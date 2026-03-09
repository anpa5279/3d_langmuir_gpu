using ThreadPinning
using MPI
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nthreads = Threads.nthreads()
mpi_pinthreads(:numa)
using Pkg
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minute, minutes, hours, seconds
## simulation parameters
Nx = 256
Ny = 256
Nz = 256
Lx = 320            # (m) domain horizontal extents
Ly = 320            # (m) domain horizontal extents
Lz = 96             # (m) domain depth 
MLD = 30.0          # m, mixed layer depth
dTdz = 0.01         # K m⁻¹, temperature gradient
alpha = 2.0e-4      # 1/K, thermal expansion coefficient
rp = 10.0           # m, radius of surface buoyancy flux
rho0 = 1025.0       # kg m⁻³, seawater density
rho_tracer = 1300.0 # kg m⁻³, reference density for tracer
T0 = 25.0           # C, temperature at the surface
u₁₀ = 5.75          # (m s⁻¹) wind speed at 10 meters above the ocean
min_step = 0.2
arch = Distributed(CPU())
# defining grid
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
@show grid

# buoyancy
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = alpha))
#beta = buoyancy.equation_of_state.haline_contraction

# BCs
u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),
                                bottom = GradientBoundaryCondition(dTdz))
g = Oceananigans.defaults.gravitational_acceleration
wp = -0.001
Sj = 0.1
area = pi*rp^2 # m², area for tracer
x_area = [Lx/2-rp, Lx/2+rp]
y_area = [Ly/2-rp, Ly/2+rp]
@inline function sflux(x, y, t) 
    if x >= x_area[1] && x <= x_area[2] && y >= y_area[1] && y <= y_area[2]
        return wp*Sj
    else
        return 0.0
    end
end
S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(sflux), 
                                bottom = GradientBoundaryCondition(0.0))

# closure
Re = 3000
w_max = 0.10747783287769483
visc = w_max*Lz/Re # 1.0e-5 # m² s⁻¹
sgs = ScalarDiffusivity(ν=visc, κ=visc)

## defining model
model = NonhydrostaticModel(grid;
                            buoyancy, 
                            advection = WENO(),
                            tracers = (:T, :S,),
                            timestepper = :RungeKutta3,
                            boundary_conditions = (u = u_bcs, v = v_bcs, S=S_bcs, T=T_bcs),
                            closure = sgs
                            )
@show model
## ICs
r(x, y, z) = (randn(Xoshiro())) * exp(z/4)
uᵢ(x, y, z) = wp * r(x, y, z)
vᵢ(x, y, z) = -wp * r(x, y, z)
Tᵢ(x, y, z) = z > - MLD ? T0 : 
                T0 + dTdz * (z + MLD)+dTdz * Lz * 1e-6 * r(x, y, z)

set!(model, u=uᵢ, v=vᵢ, T=Tᵢ, S=0.0)

# defining simulation
simulation = Simulation(model, Δt=min_step, stop_time = 12hours) 
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
simulation.callbacks[:progress] = Callback(progress, IterationInterval(500))
## updating cfl every time step
conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, diffusive_cfl = 1.0, min_Δt = min_step, max_Δt=30seconds) #ensrues cfl is updated ever iteration
## output files
output_interval = 0.2hours
u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
P_static = model.pressures.pHY′
P_dynamic = model.pressures.pNHS
simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T, S, P_static, P_dynamic),
                                                    with_halos=false,
                                                    array_type = Array{Float64},
                                                    schedule = TimeInterval(output_interval),
                                                    filename = "fields.jld2",
                                                    overwrite_existing = true)

# running the simulation
run!(simulation)