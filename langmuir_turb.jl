using Pkg
using MPI
using CUDA
using Random
using Statistics
using Printf
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BuoyancyFormulations: g_Earth
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.TimeSteppers: update_state!
using Oceananigans: UpdateStateCallsite
using Oceananigans.Fields: CenterField, FieldBoundaryConditions
import Oceananigans.BoundaryConditions: fill_halo_regions!

mutable struct Params
    Nx::Int         # number of points in each of x direction
    Ny::Int         # number of points in each of y direction
    Nz::Int         # number of points in the vertical direction
    Lx::Float64     # (m) domain horizontal extents
    Ly::Float64     # (m) domain horizontal extents
    Lz::Float64     # (m) domain depth 
    N²::Float64     # s⁻², initial and bottom buoyancy gradient
    initial_mixed_layer_depth::Float64 # m 
    Q::Float64      # W m⁻², surface heat flux. cooling is positive
    cᴾ::Float64     # J kg⁻¹ K⁻¹, specific heat capacity of seawater
    ρₒ::Float64     # kg m⁻³, average density at the surface of the world ocean
    dTdz::Float64   # K m⁻¹, temperature gradient
    T0::Float64     # C, temperature at the surface   
    β::Float64      # 1/K, thermal expansion coefficient
    u₁₀::Float64    # (m s⁻¹) wind speed at 10 meters above the ocean
    La_t::Float64   # Langmuir turbulence number
end

#defaults, these can be changed directly below 128, 128, 160, 320.0, 320.0, 96.0
p = Params(32, 32, 32, 320.0, 320.0, 96.0, 5.3e-9, 33.0, 0.0, 4200.0, 1000.0, 0.01, 17.0, 2.0e-4, 5.75, 0.3)

#referring to files with desiraed functions
include("stokes.jl")
include("smagorinsky_forcing.jl")
# Automatically distribute among available processors
arch = Distributed(GPU())

rank = arch.local_rank
Nranks = MPI.Comm_size(arch.communicator)
println("Hello from process $rank out of $Nranks")
#arch = GPU()#arch = Distributed(GPU())

grid = RectilinearGrid(arch; size=(p.Nx, p.Ny, p.Nz), extent=(p.Lx, p.Ly, p.Lz))
#stokes drift
z_d = collect(-p.Lz + grid.z.Δᵃᵃᶜ/2 : grid.z.Δᵃᵃᶜ : -grid.z.Δᵃᵃᶜ/2)
dudz = dstokes_dz(z_d, p.u₁₀)
new_dUSDdz = Field{Nothing, Nothing, Center}(grid)
set!(new_dUSDdz, reshape(dudz, 1, 1, :))

u_f = p.La_t^2 * (stokes_velocity(-grid.z.Δᵃᵃᶜ/2, p.u₁₀)[1])
τx = -(u_f^2)
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 2e-4), constant_salinity = 35.0)
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(p.Q / (p.cᴾ * p.ρₒ * p.Lx * p.Ly)),
                                bottom = GradientBoundaryCondition(p.dTdz))

#my own smagorinsky sub grid scale implementation
u_SGS = Forcing(∂ⱼ_τ₁ⱼ, discrete_form=true)
v_SGS = Forcing(∂ⱼ_τ₂ⱼ, discrete_form=true)
w_SGS = Forcing(∂ⱼ_τ₃ⱼ, discrete_form=true)
T_SGS = Forcing(∇_dot_qᶜ, discrete_form=true)

#setting up viscosity
νₑ = CenterField(grid, boundary_conditions=FieldBoundaryConditions(grid, (Center, Center, Center)))

model = NonhydrostaticModel(; grid, buoyancy, #coriolis,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = (:T),
                            closure = nothing, #closure = Smagorinsky(coefficient=0.1)
                            stokes_drift = UniformStokesDrift(∂z_uˢ=new_dUSDdz),
                            boundary_conditions = (u=u_bcs, T=T_bcs),
                            forcing = (u=u_SGS, v = v_SGS, w = w_SGS, T = T_SGS),
                            auxiliary_fields = (νₑ = νₑ,))
@show model
# random seed
r_xy(a) = randn(Xoshiro(1234), 3 * p.Nx)[Int(1 + round((p.Nx) * a/(p.Lx + grid.Δxᶜᵃᵃ)))]
r_z(z) = randn(Xoshiro(1234), p.Nz +1)[Int(1 + round((p.Nz) * z/(-p.Lz)))] * exp(z/4)
@show "rand equations made"
Tᵢ(x, y, z) = z > - p.initial_mixed_layer_depth ? p.T0 : p.T0 + p.dTdz * (z + p.initial_mixed_layer_depth)+ p.dTdz * model.grid.Lz * 1e-6 * r_z(z) * r_xy(y) * r_xy(x + p.Lx)
uᵢ(x, y, z) = u_f * 1e-1 * r_z(z) * r_xy(y) * r_xy(x + p.Lx)
wᵢ(x, y, z) = u_f * 1e-1 * r_z(z) * r_xy(y) * r_xy(x + p.Lx)
@show "equations defined"
set!(model, u=uᵢ, w=wᵢ, T=Tᵢ)
update_state!(model; compute_tendencies = true)

simulation = Simulation(model, Δt=30.0, stop_time = 96hours) #stop_time = 96hours,
@show simulation

u, v, w = model.velocities
T = model.tracers.T
@show T
νₑ = model.auxiliary_fields.νₑ

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

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

conjure_time_step_wizard!(simulation, cfl=0.5, max_Δt=30seconds)

#output files
function save_IC!(file, model)
    file["IC/friction_velocity"] = p.La_t^2 * stokes_velocity(-grid.z.Δᵃᵃᶜ/2, p.u₁₀)[1]
    file["IC/stokes_velocity"] = stokes_velocity(-grid.z.Δᵃᵃᶜ/2, p.u₁₀)[1]
    file["IC/wind_speed"] = p.u₁₀
    return nothing
end
output_interval = 60minutes

W = Average(w, dims=(1, 2))
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
T = Average(T, dims=(1, 2))

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, w, νₑ),
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "outputs/langmuir_turbulence_fields.jld2", #$(rank)
                                                      overwrite_existing = true,
                                                      init = save_IC!)
                                                      
simulation.output_writers[:averages] = JLD2OutputWriter(model, (; U, V, W, T),
                                                    schedule = AveragedTimeInterval(output_interval, window=output_interval),
                                                    filename = "outputs/langmuir_turbulence_averages.jld2",
                                                    overwrite_existing = true)

#simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(6.8e4), prefix="model_checkpoint_$(rank)")

function update_viscosity(model)
    arch = model.architecture
    u = model.velocities.u
    v = model.velocities.v
    w = model.velocities.w
    grid = model.grid
    νₑ = model.auxiliary_fields.νₑ
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    fill_halo_regions!(w)
    launch!(arch, grid, :xyz, smagorinsky_visc!, grid, u, v, w, νₑ)
    fill_halo_regions!(νₑ)
end 
@show "begin simulation"
simulation.callbacks[:visc_update] = Callback(update_viscosity, IterationInterval(1), callsite=UpdateStateCallsite())
run!(simulation) #; pickup = true
