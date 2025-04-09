using Pkg
using MPI
using CUDA
using Statistics
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BuoyancyFormulations: g_Earth

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
p = Params(32, 32, 32, 128, 128, 96.0, 5.3*(10^(-9)), 33.0, 5.0, 3991.0, 1000.0, 0.006667, 17.0, 2.0e-4, 5.75, 0.29)



# Automatically distribute among available processors
arch = Distributed(GPU())
@show arch
rank = arch.local_rank
Nranks = MPI.Comm_size(arch.communicator)
println("Hello from process $rank out of $Nranks")

grid = RectilinearGrid(arch; size=(p.Nx, p.Ny, p.Nz), extent=(p.Lx, p.Ly, p.Lz))
@show grid

#stokes drift
function stokes_kernel(f, z, u₁₀)
    α = 0.00615
    fₚ = 2π * 0.13 * g_Earth / u₁₀ # rad/s (0.22 1/s)
    return 2.0 * α * g_Earth / (fₚ * f) * exp(2.0 * f^2 * z / g_Earth - (fₚ / f)^4)
end
function stokes_velocity(z, u₁₀)
    u = Array{Float64}(undef, length(z))
    a = 0.1
    b = 5000.0
    nf = 3^9
    df = (b -  a) / nf
    for i in 1:length(z)
        σ1 = a + 0.5 * df
        u_temp = 0.0
        for k in 1:nf
            u_temp = u_temp + stokes_kernel(σ1, z[i], u₁₀)
            σ1 = σ1 + df
        end 
        u[i] = df * u_temp
    end
    return u
end
function dstokes_dz(z, u₁₀)
    dudz = Array{Float64}(undef, length(z))
    for j in 1:length(z)
        z1 = z[j]
        u1 = stokes_velocity(z1, u₁₀)[1]
        z2 = z[j] + 1e-6
        u2 = stokes_velocity(z2 + 1e-6, u₁₀)[1]
        dudz[j] = (u1 - u2) / (z1 - z2)
    end
    return dudz
end 
z_d = reverse(collect(znodes(grid, Center())))
const dudz = dstokes_dz(z_d, p.u₁₀)
@show dudz
new_dUSDdz = Field{Nothing, Nothing, Center}(grid)
function compute_new_Stokes(sim)
    
    set!(new_dUSDdz, reshape(dudz, 1, 1, :))
    
    return nothing
end 

auxiliary_fields = (; new_dUSDdz)

u_f = p.La_t^2 * (stokes_velocity(z_d[1], p.u₁₀)[1])
τx = -(u_f^2)
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
@show u_bcs

#temperature bcs
buoyancy = BuoyancyTracer()
b_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(p.N²))

#coriolis = FPlane(f=1e-4) # s⁻¹

model = NonhydrostaticModel(; grid, buoyancy, #coriolis,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = (:b),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=new_dUSDdz),
                            boundary_conditions = (u=u_bcs, b=b_bcs), auxiliary_fields,) 
@show model

# random seed
Ξ(z) = randn() * exp(z / 4)

@inline stratification(z) = z < - p.initial_mixed_layer_depth ? p.N² * z : p.N² * (-p.initial_mixed_layer_depth)

bᵢ(x, y, z) = stratification(z) + 1e-1 * Ξ(z) * p.N² * p.Lz

uᵢ(x, y, z) = u_f * 1e-1 * Ξ(z)
wᵢ(x, y, z) = u_f * 1e-1 * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, b=bᵢ)

simulation = Simulation(model, Δt=45.0, stop_time = 24hours)
@show simulation

conjure_time_step_wizard!(simulation, cfl=0.5, max_Δt=30seconds)
output_interval = 5minutes

simulation.callbacks[:Stokes] = Callback(compute_new_Stokes, IterationInterval(1))

@show model.velocities
fields_to_output = merge(model.velocities, model.tracers, (; νₑ=model.diffusivity_fields.νₑ))

simulation.output_writers[:fields] = JLD2Writer(model, fields_to_output,
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "langmuir_turbulence_fields_$rank.jld2",
                                                      overwrite_existing = true,
                                                      with_halos = false)

u, v, w = model.velocities
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
W = Average(w, dims=(1, 2))
wu = Average(w * u, dims=(1, 2))
wv = Average(w * v, dims=(1, 2))

simulation.output_writers[:averages] = JLD2Writer(model, (; U, V, W, wu, wv),
                                                        schedule = AveragedTimeInterval(output_interval, window=2minutes),
                                                        filename = "langmuir_turbulence_averages_$rank.jld2",
                                                        overwrite_existing = true,
                                                        with_halos = false)

#simulation.output_writers[:fields] = JLD2OutputWriter(model, model.forcing,
#                                                        schedule = TimeInterval(output_interval),
#                                                        filename = "forcing_$rank.jld2",
#                                                        overwrite_existing = true)

run!(simulation)
