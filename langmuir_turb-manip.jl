using Pkg
using MPI
using CUDA
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans: defaults #using Oceananigans.BuoyancyFormulations: g_Earth
using Oceananigans.DistributedComputations
const Nx = 128        # number of points in each of x direction
const Ny = 128        # number of points in each of y direction
const Nz = 128        # number of points in the vertical direction
const Lx = 320    # (m) domain horizont
const Ly = 320     # (m) domain horizontal extents
const Lz = 96      # (m) domain depth
const amplitude = 0.8      # m
const wavelength = 60.0    # m
const τx = -3.72e-5       # m² s⁻², surface kinematic momentum flux
const Jᵇ = 2.307e-8       # m² s⁻³, surface buoyancy flux
const N² = 1.936e-5       # s⁻², initial and bottom buoyancy gradient
const initial_mixed_layer_depth = 33.0  #m
# Automatically distribute among available processors
MPI.Init() # Initialize MPI
Nranks = MPI.Comm_size(MPI.COMM_WORLD)
arch = Nranks > 1 ? Distributed(GPU()) : GPU()

# Determine rank safely depending on architecture
rank = arch isa Distributed ? arch.local_rank : 0
Nranks = arch isa Distributed ? MPI.Comm_size(arch.communicator) : 1

grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)) #arch
@show grid

const g_Earth = defaults.gravitational_acceleration
const wavenumber = 2π / wavelength # m⁻¹
const frequency = sqrt(g_Earth * wavenumber) # s⁻¹

# The vertical scale over which the Stokes drift of a monochromatic surface wave
# decays away from the surface is `1/2wavenumber`, or
const vertical_scale = wavelength / 4π

# Stokes drift velocity at the surface
const Uˢ = amplitude^2 * wavenumber * frequency # m s⁻¹
@show Uˢ

@inline uˢ(z) = Uˢ * exp(z / vertical_scale)

@inline ∂z_uˢ(z, t) = 1 / vertical_scale * Uˢ * exp(z / vertical_scale)

u_boundary_conditions = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
@show u_boundary_conditions

b_boundary_conditions = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵇ),
                                                bottom = GradientBoundaryCondition(N²))
@show b_boundary_conditions

coriolis = FPlane(f=1e-4) # s⁻¹

model = NonhydrostaticModel(; grid, coriolis,
                            advection = WENO(order=5),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            boundary_conditions = (u=u_boundary_conditions, b=b_boundary_conditions))
@show model

r_z(z) = randn(Xoshiro())# * exp(z/4)
uᵢ(x, y, z) = z > - initial_mixed_layer_depth ? (u★ * r_z(z)* 1e-1) : 0.0
vᵢ(x, y, z) = -uᵢ(x, y, z)
bᵢ(x, y, z) = z > - initial_mixed_layer_depth ? (N² * (-initial_mixed_layer_depth) + N² * model.grid.Lz * 1e-1 * r_z(z)) : N² * z
set!(model, u=uᵢ, w=0.0, v=vᵢ, b=bᵢ)
simulation = Simulation(model, Δt=30.0, stop_time=240hours)
@show simulation

conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, max_Δt=30seconds)

output_interval = 2.4*hours

fields_to_output = merge(model.velocities, model.tracers, (; νₑ=model.diffusivity_fields.νₑ))

simulation.output_writers[:fields] = JLD2Writer(model, fields_to_output,
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

simulation.output_writers[:averages] = JLD2Writer(model, (; U, V, B, wu, wv),
                                                        schedule = AveragedTimeInterval(output_interval, window=2minutes),
                                                        filename = "langmuir_turbulence_averages_$rank.jld2",
                                                        overwrite_existing = true,
                                                        with_halos = false)

run!(simulation)