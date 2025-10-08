using Pkg
using MPI
using CUDA
using Random
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BuoyancyFormulations: g_Earth
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.TimeSteppers: update_state!
using Oceananigans: UpdateStateCallsite
using Oceananigans.Fields: CenterField, FieldBoundaryConditions
import Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Utils: launch!

const Nx = 128        # number of points in each of x direction
const Ny = 128        # number of points in each of y direction
const Nz = 128        # number of points in the vertical direction
const Lx = 320    # (m) domain horizontal extents
const Ly = 320    # (m) domain horizontal extents
const Lz = 96    # (m) domain depth 
const N² = 5.3e-9    # s⁻², initial and bottom buoyancy gradient
const initial_mixed_layer_depth = 30.0 # m 
const Q = 5.0     # W m⁻², surface heat flux. cooling is positive
const cᴾ = 4200.0    # J kg⁻¹ K⁻¹, specific heat capacity of seawater
const ρₒ = 1026.0    # kg m⁻³, average density at the surface of the world ocean
const dTdz = 0.01  # K m⁻¹, temperature gradient
const T0 = 25.0    # C, temperature at the surface  
const S₀ = 35.0    # ppt, salinity 
const β = 2.0e-4     # 1/K, thermal expansion coefficient
const u₁₀ = 5.75   # (m s⁻¹) wind speed at 10 meters above the ocean
const La_t = 0.3  # Langmuir turbulence number

include("smagorinsky_forcing.jl")
# Automatically distribute among available processors
MPI.Init() # Initialize MPI
Nranks = MPI.Comm_size(MPI.COMM_WORLD)
arch = Nranks > 1 ? Distributed(GPU()) : GPU()

# Determine rank safely depending on architecture
rank = arch isa Distributed ? arch.local_rank : 0
Nranks = arch isa Distributed ? MPI.Comm_size(arch.communicator) : 1

grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))

#stokes drift
include("stokes.jl")
dusdz = Field{Nothing, Nothing, Center}(grid)
z1d = grid.z.cᵃᵃᶜ[1:Nz]
dusdz_1d = dstokes_dz.(z1d, u₁₀)
set!(dusdz, dusdz_1d)
us_1d = stokes_velocity.(z1d, u₁₀)
stokes = UniformStokesDrift(∂z_uˢ=dusdz)

#BCs
us_top = us_1d[Nz]
u_f = La_t^2 * us_top
τx = -(u_f^2)
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx), bottom = GradientBoundaryCondition(0.0)) 
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q / (cᴾ * ρₒ * Lx * Ly)),
                                bottom = GradientBoundaryCondition(dTdz))

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 2e-4), constant_salinity = 35.0)
coriolis = FPlane(f=1e-4) # s⁻¹

#my own smagorinsky sub grid scale implementation
u_SGS = Forcing(∂ⱼ_τ₁ⱼ, discrete_form=true)
v_SGS = Forcing(∂ⱼ_τ₂ⱼ, discrete_form=true)
w_SGS = Forcing(∂ⱼ_τ₃ⱼ, discrete_form=true)
T_SGS = Forcing(∇_dot_qᶜ, discrete_form=true)

#setting up viscosity
νₑ = CenterField(grid)

model = NonhydrostaticModel(; grid, buoyancy, coriolis,
                            advection = WENO(),
                            tracers = (:T,),
                            timestepper = :RungeKutta3,
                            closure = nothing, #closure = Smagorinsky(coefficient=0.1)
                            stokes_drift = stokes,
                            boundary_conditions = (u=u_bcs, T=T_bcs),
                            forcing = (u=u_SGS, v = v_SGS, w = w_SGS, T = T_SGS),
                            auxiliary_fields = (νₑ = νₑ,))
@show model
## ICs
r(x, y, z) = randn(Xoshiro(1234), (Nx + Ny +Nz+3))[Int(1 + round(Nx*x/Lx+Ny*y/Ly-Nz*z/Lz))] * exp(z / 4)
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+dTdz * model.grid.Lz * 1e-6 * r(x, y, z)
uᵢ(x, y, z) = u_f * r(x, y, z)
set!(model, u=uᵢ, T=Tᵢ)

simulation = Simulation(model, Δt=30.0, stop_time = 24hours) #stop_time = 96hours,
@show simulation

function progress(simulation)
    u, v, w = simulation.model.velocities
    T = model.tracers.T
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

conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, max_Δt=30seconds)

#output files
function save_IC!(file, model)
    file["IC/friction_velocity"] = La_t^2 * stokes_velocity(-grid.z.Δᵃᵃᶜ/2, u₁₀)[1]
    file["IC/stokes_velocity"] = stokes_velocity(-grid.z.Δᵃᵃᶜ/2, u₁₀)[1]
    file["IC/wind_speed"] = u₁₀
    return nothing
end

u, v, w = model.velocities
T = model.tracers.T
νₑ = model.auxiliary_fields.νₑ

output_interval = 60minutes
dir = "forcing-function/"
simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, νₑ, T),
                                                      dir = dir,
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "forcing_fields_async_true.jld2", #$(rank)
                                                      array_type = Array{Float64},
                                                      overwrite_existing = true,
                                                      init = save_IC!)

function update_viscosity(model)
    arch = model.architecture
    u = model.velocities.u
    v = model.velocities.v
    w = model.velocities.w
    grid = model.grid
    νₑ = model.auxiliary_fields.νₑ

    # Ensure halo regions are filled and GPU is synchronized
    fill_halo_regions!(νₑ)
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    fill_halo_regions!(w)

    # Synchronize GPU operations
    synchronize()

    launch!(arch, grid, :xyz, smagorinsky_visc!, grid, u, v, w, νₑ)
    fill_halo_regions!(νₑ)
    return nothing
end
visc_callback = Callback(update_viscosity, IterationInterval(1), callsite=UpdateStateCallsite())
simulation.callbacks[:visc_update] = visc_callback
update_state!(model, [visc_callback,]; compute_tendencies = false)
@show "after update_state"
@show νₑ
@show "begin simulation"
run!(simulation) #; pickup = true