using Pkg
using MPI
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.Units: minute, minutes, hours
using Oceananigans.BuoyancyFormulations: g_Earth

mutable struct Params
    Nx::Int
    Ny::Int     # number of points in each of horizontal directions
    Nz::Int     # number of points in the vertical direction
    Lx::Float64
    Ly::Float64     # (m) domain horizontal extents
    Lz::Float64     # (m) domain depth 
    amplitude::Float64 # m
    wavelength::Float64 # m
    τx::Float64 # m² s⁻², surface kinematic momentum flux
    Jᵇ::Float64 # m² s⁻³, surface buoyancy flux
    N²::Float64 # s⁻², initial and bottom buoyancy gradient
    initial_mixed_layer_depth::Float64 #m
end

#defaults, these can be changed directly below
params = Params(32, 32, 32, 128.0, 128.0, 64.0, 0.8, 60.0, -3.72e-5, 2.307e-8, 1.936e-5, 33.0)

# Automatically distributes among available processors

arch = Distributed(GPU())
@show arch
rank = arch.local_rank
Nranks = MPI.Comm_size(arch.communicator)
println("Hello from process $rank out of $Nranks")

grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), extent=(params.Lx, params.Ly, params.Lz))
@show grid

const wavenumber = 2π / params.wavelength # m⁻¹
const frequency = sqrt(g_Earth * wavenumber) # s⁻¹
@show wavenumber, frequency

# The vertical scale over which the Stokes drift of a monochromatic surface wave
# decays away from the surface is `1/2wavenumber`, or
const vertical_scale = params.wavelength / 4π
@show vertical_scale

# Stokes drift velocity at the surface
const Uˢ = params.amplitude^2 * wavenumber * frequency # m s⁻¹
@show Uˢ

@inline uˢ(z) = Uˢ * exp(z / vertical_scale)

@inline ∂z_uˢ(z, t) = 1 / vertical_scale * Uˢ * exp(z / vertical_scale)

u_boundary_conditions = FieldBoundaryConditions(top = FluxBoundaryCondition(params.τx))
@show u_boundary_conditions

b_boundary_conditions = FieldBoundaryConditions(top = FluxBoundaryCondition(params.Jᵇ),
                                                bottom = GradientBoundaryCondition(params.N²))
@show b_boundary_conditions

coriolis = FPlane(f=1e-4) # s⁻¹

model = NonhydrostaticModel(; grid, coriolis,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            boundary_conditions = (u=u_boundary_conditions, b=b_boundary_conditions))
@show model

@inline Ξ(z) = randn() * exp(z / 4)

@inline stratification(z) = z < - params.initial_mixed_layer_depth ? params.N² * z : params.N² * (-params.initial_mixed_layer_depth)

@inline bᵢ(x, y, z) = stratification(z) + 1e-1 * Ξ(z) * params.N² * model.grid.Lz

u★ = sqrt(abs(params.τx))
@inline uᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
@inline wᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, b=bᵢ)

simulation = Simulation(model, Δt=45.0, stop_time=4hours)
@show simulation

conjure_time_step_wizard!(simulation, cfl=1.0, max_Δt=1minute)

output_interval = 5minutes

fields_to_output = merge(model.velocities, model.tracers, (; νₑ=model.diffusivity_fields.νₑ))

simulation.output_writers[:fields] = JLD2OutputWriter(model, fields_to_output,
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "langmuir_turbulence_fields_$rank.jld2",
                                                      overwrite_existing = true,
                                                      with_halos = false)

u, v, w = model.velocities
b = model.tracers.b

U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
B = Average(b, dims=(1, 2))
wu = Average(w * u, dims=(1, 2))
wv = Average(w * v, dims=(1, 2))

simulation.output_writers[:averages] = JLD2OutputWriter(model, (; U, V, B, wu, wv),
                                                        schedule = AveragedTimeInterval(output_interval, window=2minutes),
                                                        filename = "langmuir_turbulence_averages_$rank.jld2",
                                                        overwrite_existing = true,
                                                        with_halos = false)

run!(simulation)