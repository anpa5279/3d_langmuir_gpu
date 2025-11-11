using Pkg
using MPI
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
T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),#FluxBoundaryCondition(Q / (cᴾ * ρₒ * Lx * Ly)),
                                bottom = GradientBoundaryCondition(0.0))
@show "Boundary conditions set"
coriolis = FPlane(f=1e-4) # s⁻¹
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = β), constant_salinity = S0)
@show "Additional model parameters set"
model = NonhydrostaticModel(; grid, buoyancy, coriolis,
                            advection = WENO(),
                            tracers = (:BOH3, :BOH4, :CO2, :CO3, :HCO3, :OH, :T),
                            timestepper = :CCRungeKutta3, #chemical kinetics are embedded inthis timestepper
                            closure = Smagorinsky(coefficient=0.1),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=dusdz),
                            boundary_conditions = (u=u_bcs, T=T_bcs))#, CO2 = DIC_bcs)) 
@show model

# random seed
r_xy(a) = randn(Xoshiro(1234), 3 * Nx)[Int(1 + round((Nx) * a/(Lx + grid.Δxᶜᵃᵃ)))]
r_z(z) = randn(Xoshiro(1234), Nz +1)[Int(1 + round((Nz) * z/(-Lz)))] * exp(z/4)
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+ dTdz * model.grid.Lz * 1e-6 * r_z(z) * r_xy(y) * r_xy(x + Lx)
uᵢ(x, y, z) = u_f * 1e-1 * r_z(z) * r_xy(y) * r_xy(x + Lx)
perturb = 1e3
set!(model, BOH3 = 2.97e2, BOH4 = 1.19e2, CO2 = 7.57e0 * perturb, CO3 = 3.15e2, HCO3 = 1.67e3, OH = 9.6e0, u=uᵢ, T=Tᵢ)

day = 24hours
simulation = Simulation(model, Δt=30, stop_time = 1*day)

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
                   mean(simulation.model.tracers.CO2),
                   mean(simulation.model.tracers.CO3),
                   mean(simulation.model.tracers.HCO3),
                   mean(simulation.model.tracers.OH),
                   mean(simulation.model.tracers.BOH3),
                   mean(simulation.model.tracers.BOH4))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, max_Δt=30seconds)
#output files
function save_IC!(file, model)
    if rank == 0
        file["IC/friction_velocity"] = u_f
        file["IC/stokes_velocity"] = stokes_velocity(-grid.z.Δᵃᵃᶜ/2, u₁₀)[1]
        file["IC/wind_speed"] = u₁₀
    end
    return nothing
end

output_interval = 2hours

outputs_fields = merge(simulation.model.velocities, simulation.model.tracers)

simulation.output_writers[:fields] = JLD2Writer(model, outputs_fields,
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "langmuir_turbulence_fields.jld2", #$(rank)
                                                      overwrite_existing = true,
                                                      init = save_IC!)

#simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(1*day), prefix="model_checkpoint")

run!(simulation)#; pickup = true)
