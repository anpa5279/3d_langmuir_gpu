using Pkg
using MPI
using CUDA
using JLD2
using Statistics
using Printf
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BuoyancyFormulations: g_Earth
using Oceananigans.Fields: interior, set!
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

p = Params(128, 128, 160, 320.0, 320.0, 96.0, 5.3e-9, 33.0, 0.0, 4200.0, 1000.0, 0.01, 17.0, 2.0e-4, 5.75, 0.3)

# Automatically distribute among available processors
arch = Distributed(GPU())
@show arch
rank = arch.local_rank
Nranks = MPI.Comm_size(arch.communicator)
println("Hello from process $rank out of $Nranks")

grid = RectilinearGrid(arch; size=(p.Nx, p.Ny, p.Nz), extent=(p.Lx, p.Ly, p.Lz))
@show grid

function stokes_velocity(z, u₁₀)
    u = Array{Float64}(undef, length(z))
    α = 0.00615
    fₚ = 2π * 0.13 * g_Earth / u₁₀ # rad/s (0.22 1/s)
    a = 0.1
    b = 5000.0
    nf = 3^9
    df = (b -  a) / nf
    for i in 1:length(z)
        σ = a + 0.5 * df
        u_temp = 0.0
        for k in 1:nf
            u_temp = u_temp + (2.0 * α * g_Earth / (fₚ * σ) * exp(2.0 * σ^2 * z[i] / g_Earth - (fₚ / σ)^4))
            σ = σ + df
        end
        u[i] = df * u_temp
    end
    return u
end
function dstokes_dz(z, u₁₀)
    dudz = Array{Float64}(undef, length(z))
    α = 0.00615
    fₚ = 2π * 0.13 * g_Earth / u₁₀ # rad/s (0.22 1/s)
    a = 0.1
    b = 5000.0
    nf = 3^9
    df = (b -  a) / nf
    for i in 1:length(z)
        σ = a + 0.5 * df
        du_temp = 0.0
        for k in 1:nf
            du_temp = du_temp + (4.0 * α * σ/ (fₚ) * exp(2.0 * σ^2 * z[i] / g_Earth - (fₚ / σ)^4))
            σ = σ + df
        end
        dudz[i] = df * du_temp
    end
    return dudz
end

const z_d = collect(reverse(-p.Lz + grid.z.Δᵃᵃᶜ/2 : grid.z.Δᵃᵃᶜ : -grid.z.Δᵃᵃᶜ/2))
const dudz = dstokes_dz(z_d, p.u₁₀)
new_dUSDdz = Field{Nothing, Nothing, Center}(grid)
set!(new_dUSDdz, reshape(dudz, 1, 1, :))

u_f = p.La_t^2 * (stokes_velocity(-grid.z.Δᵃᵃᶜ/2, p.u₁₀)[1])
τx = -(u_f^2)
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
@show u_bcs

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 2e-4), constant_salinity = 35.0)
@show buoyancy
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(p.Q / (p.cᴾ * p.ρₒ * p.Lx * p.Ly)),
                                bottom = GradientBoundaryCondition(p.dTdz))
##coriolis = FPlane(f=1e-4) # s⁻¹

model = NonhydrostaticModel(; grid, buoyancy, #coriolis,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = (:T),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=new_dUSDdz),
                            boundary_conditions = (u=u_bcs, T=T_bcs))

simulation = Simulation(model, Δt=30.0, stop_time = 96hours)
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

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))
output_interval = 10minutes

fields_to_output = merge(model.velocities, model.tracers)

simulation.output_writers[:fields] = JLD2Writer(model, fields_to_output,
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "langmuir_turbulence_fields_$(rank).jld2",
                                                      overwrite_existing = false)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(Int(5.0e4)), prefix="model_checkpoint_$(rank)")
# Load the checkpoint file manually
file = jldopen("model_checkpoint_$(rank)_iteration50000.jld2", "r")
t = file["NonhydrostaticModel/clock"]

# Now manually set them in your simulation
simulation.model.clock.time = t.time
simulation.model.clock.iteration = t.iteration

u_full = Array(file["NonhydrostaticModel/u/data"])
v_full = Array(file["NonhydrostaticModel/v/data"])
w_full = Array(file["NonhydrostaticModel/w/data"])
T_full = Array(file["NonhydrostaticModel/T/data"])

u_interior = u_full[grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx]
v_interior = v_full[grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx]
w_interior = w_full[grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx]
T_interior = T_full[grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx, grid.Hx+1:end - grid.Hx]

set!(simulation.model.velocities.u, CuArray(reshape(u_interior, :)))
set!(simulation.model.velocities.v, CuArray(reshape(v_interior, :)))
set!(simulation.model.velocities.w, CuArray(reshape(w_interior, :)))
set!(simulation.model.tracers.T, CuArray(reshape(T_interior, :)))
close(file)

simulation.Δt = 30.0
run!(simulation) #; pickup = true) #"model_checkpoint_$(rank)_iteration50000.jld2") to run with the file, comment out the file you read in