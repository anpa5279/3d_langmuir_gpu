#using ThreadPinning
#using MPI
#MPI.Init()
#rank = MPI.Comm_rank(MPI.COMM_WORLD)
#nthreads = Threads.nthreads()
#mpi_pinthreads(:numa)
using Pkg
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BoundaryConditions: fill_halo_regions!, OpenBoundaryCondition
using Oceananigans.Models: BoundaryAdjacentMean
using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ

using Logging
global_logger(SimpleLogger(stdout, Logging.Info))

## simulation parameters
Nx = 8
Ny = 8
Nz = 8
Lx = 320    # (m) domain horizontal extents
Ly = 320    # (m) domain horizontal extents
Lz = 96    # (m) domain depth 
initial_mixed_layer_depth = 30.0 # m 
dTdz = 0.01  # K m⁻¹, temperature gradient
β = 2.0e-4     # 1/K, thermal expansion coefficient
w_max = 0.10747783287769483
La_t = 0.3  # Langmuir turbulence number

output_interval = 0.2hours
# grid 
arch = CPU() #Distributed(CPU())
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
@show grid
# stokes 
g = Oceananigans.defaults.gravitational_acceleration
amplitude = 0.8 # m
wavelength = 60  # m
wavenumber = 2π / wavelength # m⁻¹
frequency = sqrt(g * wavenumber) # s⁻¹

# The vertical scale over which the Stokes drift of a monochromatic surface wave
# decays away from the surface is `1/2wavenumber`, or
const vertical_scale = wavelength / 4π

# Stokes drift velocity at the surface
const Uˢ = amplitude^2 * wavenumber * frequency # m s⁻¹
uˢ(z) = Uˢ * exp(z / vertical_scale)
∂z_uˢ(z, t) = 1 / vertical_scale * Uˢ * exp(z / vertical_scale)

# BCs
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))

u_f = La_t^2 * uˢ(0) # friction velocity scale, m s⁻¹
b0 = -4e-1 # m s⁻²
Jᵇ = -u_f*b0 # m² s⁻³, surface buoyancy flux
@inline function bflux_t(x, y, t) 
    if (t <= 6hours)
        σ = 10.0 # m
        return Jᵇ/(2*pi* σ^2) * exp(-(x-Lx/2)^2 / (2 * σ^2)) * exp(-(y-Ly/2)^2 / (2 * σ^2)) 
    else
        return 0.0
    end
end
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(bflux_t), 
                                    bottom = GradientBoundaryCondition(g*β*dTdz))

buoyancy = BuoyancyTracer()
## ICs
r(x, y, z) = (randn(Xoshiro())) * exp(z/4)
vᵢ(x, y, z) = -u_f * r(x, y, z)

bᵢ(x, y, z) = z > - initial_mixed_layer_depth ? g*β*dTdz * Lz * 1e-6 * r(x, y, z) : #random noise in the mixed layer
                g*β*dTdz * (z + initial_mixed_layer_depth) + g*β*dTdz * Lz * 1e-6 * r(x, y, z)

# closure
Re = 3000
visc = w_max*Lz/Re 
sgs = ScalarDiffusivity(ν=visc, κ=visc)

paths = ["updated friction velocity", "with coriolis", "with coriolis and wind stress", "with coriolis and wind stress and stokes drift", "with wind stress and stokes drift"]

## progress function
function progress(simulation)
    u, v, w = simulation.model.velocities
    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                iteration(simulation),
                prettytime(time(simulation)),
                prettytime(simulation.Δt),
                maximum(abs, u), maximum(abs, v), maximum(abs, w),
                prettytime(simulation.run_wall_time))
    @info msg
    return nothing
end

for path in paths
    if occursin("wind stress", path)
        τx = -(u_f^2)
        u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx), 
                bottom = GradientBoundaryCondition(0.0))
    else
        u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                bottom = GradientBoundaryCondition(0.0))
    end

    if occursin("stokes drift", path)
        stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ)
        global uᵢ(x, y, z) = u_f * r(x, y, z) + uˢ(z)
    else
        stokes_drift = nothing
        global uᵢ(x, y, z) = u_f * r(x, y, z)
    end

    if occursin("with coriolis", path)
        coriolis = FPlane(f=1e-4) # s⁻¹
    else
        coriolis = nothing
    end

    model = NonhydrostaticModel(grid;
                            buoyancy, coriolis, stokes_drift, 
                            advection = WENO(),
                            tracers = (:b,),
                            timestepper = :RungeKutta3,
                            boundary_conditions = (u = u_bcs, v = v_bcs, b=b_bcs),
                            closure = sgs
                            )
    @show model
    flush(stdout)

    set!(model, u=uᵢ, v=vᵢ, b=bᵢ)

    # defining simulation
    simulation = Simulation(model, Δt=30, stop_time = 12hours, wall_time_limit = 4hours) 
    @show simulation
    flush(stdout)
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(500))
    ## updating cfl every time step
    conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, min_Δt = 1.0, max_Δt=30seconds) #ensrues cfl is updated ever iteration
    ## output files
    rel_path = "$path/"
    u, v, w = model.velocities
    b = model.tracers.b
    P_static = model.pressures.pHY′ #hydrostatic pressure, (pNHS=nonhydrostatic_pressure, pHY′=hydrostatic_pressure_anomaly)
    P_dynamic = model.pressures.pNHS #nonhydrostatic pressure, (pNHS=nonhydrostatic_pressure, pHY′=hydrostatic_pressure_anomaly)
    simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, b, P_static, P_dynamic),
                                                        dir = rel_path,  with_halos=false,
                                                        array_type = Array{Float64},
                                                        schedule = TimeInterval(output_interval),
                                                        filename = "fields.jld2", 
                                                        overwrite_existing = true)

    # running the simulation
    run!(simulation)
end