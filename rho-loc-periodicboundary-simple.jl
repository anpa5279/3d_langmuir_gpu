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
using Oceananigans.TurbulenceClosures: Smagorinsky
## simulation parameters
Nx = 32        # number of points in each of x direction
Ny = 32        # number of points in each of y direction
Nz = 64        # number of points in the vertical direction
Lx = 320    # (m) domain horizontal extents
Ly = 320    # (m) domain horizontal extents
Lz = 96    # (m) domain depth 
initial_mixed_layer_depth = 30.0 # m 
Q = 5.0     # W m⁻², surface heat flux. cooling is positive
cᴾ = 4200.0    # J kg⁻¹ K⁻¹, specific heat capacity of seawater
ρₒ = 1026.0    # kg m⁻³, average density at the surface of the world ocean
dTdz = 0.01  # K m⁻¹, temperature gradient
T0 = 25.0    # C, temperature at the surface 
β = 2.0e-4     # 1/K, thermal expansion coefficient
u₁₀ = 5.75   # (m s⁻¹) wind speed at 10 meters above the ocean
La_t = 0.3  # Langmuir turbulence number

## referring to files with desiraed functions
grid = RectilinearGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)) #arch

# BCs
u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), #FluxBoundaryCondition(τx), 
                                bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
@inline function bflux_t(x, y, t) 
    if (t <= 6hours)
        σ = 10.0 # m
        c0 = mass/(molar_calcite*(Lx/Nx)*(Ly/Ny)*(Lz/Nz)) # mol/m3
        return c0/sqrt(2*pi* σ^2) * exp(-(x-Lx/2)^2 / (2 * σ^2)) * exp(-(y-Ly/2)^2 / (2 * σ^2)) 
    else
        return 0.0
    end
end
b_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(bflux_t), 
                                    bottom = GradientBoundaryCondition(dTdz))

buoyancy = BuoyancyTracer()

## defining model
model = NonhydrostaticModel(; grid,
                            buoyancy, 
                            advection = WENO(),
                            tracers = (:b,),
                            timestepper = :RungeKutta3,
                            boundary_conditions = (u = u_bcs, v = v_bcs, b=b_bcs),
                            )
@show model
## ICs
u_f = 0.001
r(x, y, z) = (1+randn(Xoshiro())) * exp(z/4)
bᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+dTdz * model.grid.Lz * 1e-6 * r(x, y, z)
uᵢ(x, y, z) = u_f * r(x, y, z)
vᵢ(x, y, z) = -u_f * r(x, y, z)

σ = 10.0 # m
c0 = mass/(molar_calcite*(Lx/Nx)*(Ly/Ny)*(Lz/Nz)) # mol/m3
CaCO3ᵢ(x, y, z) = c0/sqrt(2*pi* σ^2) * exp(-z^2 / (2 * σ^2)) * exp(-(x-Lx/2)^2 / (2 * σ^2)) * exp(-(y-Ly/2)^2 / (2 * σ^2)) 

set!(model, u=uᵢ, v=vᵢ, b=bᵢ)

# defining simulation
simulation = Simulation(model, Δt=30, stop_time = 2.0*24hours) 
@show simulation
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
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
## updating cfl every time step
conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, min_Δt = 1.0, max_Δt=30seconds) #ensrues cfl is updated ever iteration
## output files
output_interval = 0.25hours
path = "localoutputs/no tracer for NBP/simple case/"
u, v, w = model.velocities
b = model.tracers.b
CaCO3 = model.tracers.CaCO3
P_static = model.pressures.pHY′
P_dynamic = model.pressures.pNHS
simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, b, P_static, P_dynamic),
                                                    dir = path,  with_halos=false,
                                                    array_type = Array{Float64},
                                                    schedule = TimeInterval(output_interval),
                                                    filename = "NBP_fields.jld2", #$(rank)
                                                    overwrite_existing = true,
                                                    init = save_IC!)

# running the simulation
run!(simulation)
