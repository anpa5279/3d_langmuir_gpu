using Pkg
using MPI
using CUDA
using Statistics
using Printf
using Random
Pkg.develop(path="/glade/work/apauls/personal-oceananigans/Oceananigans.jl-main")
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BuoyancyFormulations: g_Earth
using Oceananigans.DistributedComputations
#include("cc.jl")
#using .CC #: CarbonateChemistry #local module
#include("strang-rk3.jl") #local module
#using .SRK3
const Nx = 128        # number of points in each of x direction
const Ny = 128        # number of points in each of y direction
const Nz = 128        # number of points in the vertical direction
const Lx = 320    # (m) domain horizontal extents
const Ly = 320    # (m) domain horizontal extents
const Lz = 96    # (m) domain depth 
const N² = 5.3e-9    # s⁻², initial and bottom buoyancy gradient
const initial_mixed_layer_depth = 30.0 # m 
const Q = 0.0     # W m⁻², surface heat flux. cooling is positive
const cᴾ = 4200.0    # J kg⁻¹ K⁻¹, specific heat capacity of seawater
const ρₒ = 1026.0    # kg m⁻³, average density at the surface of the world ocean
const dTdz = 0.01  # K m⁻¹, temperature gradient
const T0 = 25.0    # C, temperature at the surface  
const S0 = 35.0    # ppt, salinity 
const β = 2.0e-4     # 1/K, thermal expansion coefficient
const u₁₀ = 5.75   # (m s⁻¹) wind speed at 10 meters above the ocean
const La_t = 0.3  # Langmuir turbulence number

#referring to files with desiraed functions
include("stokes.jl")

# Automatically distribute among available processors
MPI.Init() # Initialize MPI
Nranks = MPI.Comm_size(MPI.COMM_WORLD)
arch = Nranks > 1 ? Distributed(CPU()) : CPU()

# Determine rank safely depending on architecture
rank = arch isa Distributed ? arch.local_rank : 0
Nranks = arch isa Distributed ? MPI.Comm_size(arch.communicator) : 1

grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)) 

#stokes drift
g_Earth = defaults.gravitational_acceleration
include("stokes.jl")
dusdz = Field{Nothing, Nothing, Center}(grid)
z_d = collect(-Lz + grid.z.Δᵃᵃᶜ/2 : grid.z.Δᵃᵃᶜ : -grid.z.Δᵃᵃᶜ/2)
dusdz_1d = dstokes_dz.(z_d, u₁₀)
set!(dusdz, reshape(dusdz_1d, 1, 1, :))
@show dusdz

#BCs
us = stokes_velocity(z_d, u₁₀)
u_f = La_t^2 * us[end]
τx = -(u_f^2)# m² s⁻², surface kinematic momentum flux
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx), 
                                bottom = GradientBoundaryCondition(0.0)) 
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), bottom = GradientBoundaryCondition(0.0))

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = β), constant_salinity = S0)

T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q / (cᴾ * ρₒ * Lx * Ly)),
                                bottom = GradientBoundaryCondition(dTdz))
#coriolis = FPlane(f=1e-4) # s⁻¹

model = NonhydrostaticModel(; grid, buoyancy, coriolis,
                            advection = WENO(),
                            tracers = (:BOH₃, :BOH₄, :CO₂, :CO₃, :HCO₃, :OH, :T, :S),
                            timestepper = :CCRungeKutta3, #chemical kinetics are embedded in this timestepper
                            closure = Smagorinsky(coefficient=0.1),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=dusdz),
                            boundary_conditions = (u=u_bcs, T=T_bcs))
@show model

# ICs
r_z(z) = randn(Xoshiro())# * exp(z/4)
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? (T0 + dTdz * model.grid.Lz * 1e-6 * r_z(z)) : T0 + dTdz * (z + initial_mixed_layer_depth) 
uᵢ(x, y, z) = z > - initial_mixed_layer_depth ? (u_f * r_z(z)) : 0.0
vᵢ(x, y, z) = -uᵢ(x, y, z)
perturb = 1e3
set!(model, u=uᵢ, w=0.0, v=vᵢ, T=Tᵢ, BOH₃ = 2.97e2, BOH₄ = 1.19e2, CO₂ = 7.57e0 * perturb, CO₃ = 3.15e2, HCO₃ = 1.67e3, OH = 9.6e0) #u=uᵢ, w=wᵢ, 

day = 24hours
simulation = Simulation(model, Δt=30, stop_time = 1hours) #stop_time = 96hours,
@show simulation

function progress(simulation)
    u, v, w = simulation.model.velocities

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n
    co2 = %.1e, co3 = %.1e, hco3 = %.1e, oh = %.1e, boh3 = %.1e, boh4 = %.1e",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(simulation.run_wall_time), 
                   mean(simulation.model.tracers.CO₂),
                   mean(simulation.model.tracers.CO₃),
                   mean(simulation.model.tracers.HCO₃),
                   mean(simulation.model.tracers.OH),
                   mean(simulation.model.tracers.BOH₃),
                   mean(simulation.model.tracers.BOH₄))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))

conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, max_Δt=30seconds)
#output files
function save_IC!(file, model)
    if rank == 0 || Nranks == 1
        file["IC/friction_velocity"] = u_f
        file["IC/stokes_velocity"] = us
        file["IC/wind_speed"] = u₁₀
    end
    return nothing
end

output_interval = 2hours

u, v, w = model.velocities
BOH₃ = model.tracers.BOH₃
BOH₄ = model.tracers.BOH₄
CO₂ = model.tracers.CO₂
CO₃ = model.tracers.CO₃
HCO₃ = model.tracers.HCO₃
OH = model.tracers.OH
T = model.tracers.T

simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T, BOH₃, BOH₄, CO₂, CO₃, HCO₃, OH),
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "langmuir_turbulence_fields.jld2", #$(rank)
                                                      overwrite_existing = true,
                                                      init = save_IC!)
#simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(30000), prefix="model_checkpoint")
run!(simulation)#; pickup = true)