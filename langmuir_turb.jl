using Pkg
using MPI
using CUDA
using Statistics
using Printf
using Random
Pkg.develop(path="/glade/work/apauls/.julia/dev/Oceananigans.jl-main")
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
arch = Nranks > 1 ? Distributed(GPU()) : GPU()

# Determine rank safely depending on architecture
rank = arch isa Distributed ? arch.local_rank : 0
Nranks = arch isa Distributed ? MPI.Comm_size(arch.communicator) : 1

#grid = RectilinearGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))

#stokes drift
dusdz = Field{Nothing, Nothing, Center}(grid)
Nx_local, Ny_local, Nz_local = size(dusdz)
z1d = grid.z.cᵃᵃᶜ[1:Nz_local]
dusdz_1d = dstokes_dz.(z1d, u₁₀)
set!(dusdz, dusdz_1d)
us = Field{Nothing, Nothing, Center}(grid)
us_1d = stokes_velocity.(z1d, u₁₀)
set!(us, us_1d)
@show dusdz

u_f = La_t^2 * (stokes_velocity(-grid.z.Δᵃᵃᶜ/2, u₁₀)[1])
τx = -(u_f^2)
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx), bottom = GradientBoundaryCondition(0.0)) 
#v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), bottom = GradientBoundaryCondition(0.0))
#w_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), bottom = ValueBoundaryCondition(0.0)) 

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = β), constant_salinity = S0)

T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),#FluxBoundaryCondition(Q / (cᴾ * ρₒ * Lx * Ly)),
                                bottom = GradientBoundaryCondition(0.0))
coriolis = FPlane(f=1e-4) # s⁻¹
#biogeochemistry = CarbonateChemistry(; grid, scale_negatives = true)
#DIC_bcs = FieldBoundaryConditions(top = GasExchange(; gas = :CO₂, temperature = (args...) -> T0, salinity = (args...) -> 35))

model = NonhydrostaticModel(; grid, buoyancy, coriolis,
                            advection = WENO(),
                            tracers = (:BOH₃, :BOH₄, :CO₂, :CO₃, :HCO₃, :OH, :T, :S),
                            timestepper = :CCRungeKutta3, #chemical kinetics are embedded inthis timestepper
                            closure = Smagorinsky(coefficient=0.1),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=dusdz),
                            boundary_conditions = (u=u_bcs, T=T_bcs))#, CO₂ = DIC_bcs)) 
#model = NonhydrostaticModel(; grid, advection = WENO(), timestepper = :RungeKutta3, stokes_drift = UniformStokesDrift(grid; ∂z_uˢ=dusdz))
@show model

# random seed
r_xy(a) = randn(Xoshiro(1234), 3 * Nx)[Int(1 + round((Nx) * a/(Lx + grid.Δxᶜᵃᵃ)))]
r_z(z) = randn(Xoshiro(1234), Nz +1)[Int(1 + round((Nz) * z/(-Lz)))] * exp(z/4)
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+ dTdz * model.grid.Lz * 1e-6 * r_z(z) * r_xy(y) * r_xy(x + Lx)
#uᵢ(x, y, z) = u_f * 1e-1 * r_z(z) * r_xy(y) * r_xy(x + Lx)
#wᵢ(x, y, z) = u_f * 1e-1 * r_z(z) * r_xy(y) * r_xy(x + Lx)
perturb = 1e3
#set!(model, u=uᵢ, w=wᵢ, N=1, P=1, Z=1, D=1, T=Tᵢ, S = 35)
set!(model, BOH₃ = 2.97e2, BOH₄ = 1.19e2, CO₂ = 7.57e0 * perturb, CO₃ = 3.15e2, HCO₃ = 1.67e3, OH = 9.6e0, T=Tᵢ, S = S0) #u=uᵢ, w=wᵢ, 

day = 24hours
simulation = Simulation(model, Δt=30, stop_time = 7*day) #stop_time = 96hours,
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

conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, max_Δt=30seconds)
#output files
function save_IC!(file, model)
    #if rank == 0
    file["IC/friction_velocity"] = u_f
    file["IC/stokes_velocity"] = stokes_velocity(-grid.z.Δᵃᵃᶜ/2, u₁₀)[1]
    file["IC/wind_speed"] = u₁₀
    #end
    return nothing
end

output_interval = 6hours

u, v, w = model.velocities
#N = model.tracers.N
#P = model.tracers.P
#Z = model.tracers.Z
#D = model.tracers.D
BOH₃ = model.tracers.BOH₃
BOH₄ = model.tracers.BOH₄
CO₂ = model.tracers.CO₂
CO₃ = model.tracers.CO₃
HCO₃ = model.tracers.HCO₃
OH = model.tracers.OH
T = model.tracers.T
W = Average(w, dims=(1, 2))
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
T = Average(T, dims=(1, 2))

simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, BOH₃, BOH₄, CO₂, CO₃, HCO₃, OH),
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "langmuir_turbulence_fields.jld2", #$(rank)
                                                      overwrite_existing = true,
                                                      init = save_IC!)
                                                      
simulation.output_writers[:averages] = JLD2Writer(model, (; U, V, W, T),
                                                    schedule = AveragedTimeInterval(output_interval, window=output_interval),
                                                    filename = "langmuir_turbulence_averages.jld2",
                                                    overwrite_existing = true)
#simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(30000), prefix="model_checkpoint")

run!(simulation)#; pickup = true)
