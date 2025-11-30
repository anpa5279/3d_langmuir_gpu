using Pkg
using MPI
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans: defaults #using Oceananigans.BuoyancyFormulations: g_Earth
using Oceananigans.DistributedComputations
using Logging
using BenchmarksTools
const N = 128        # number of points in each of x direction
const Lx = 320    # (m) domain horizont
const Ly = 320     # (m) domain horizontal extents
const Lz = 96      # (m) domain depth
const amplitude = 0.8      # m
const wavelength = 60.0    # m
const τx = -3.72e-5       # m² s⁻², surface kinematic momentum flux
const Jᵇ = 2.307e-8       # m² s⁻³, surface buoyancy flux
const N² = 1.936e-5       # s⁻², initial and bottom buoyancy gradient
const initial_mixed_layer_depth = 33.0  #m

MPI.Init()

      comm = MPI.COMM_WORLD
local_rank = MPI.Comm_rank(comm)
         R = MPI.Comm_size(comm)

 Nx = parse(Int, N)
 Ny = parse(Int, N)
 Nz = parse(Int, N)
 Rx = parse(Int, 1)
 Ry = parse(Int, 2)
 Rz = parse(Int, 1)

@assert Rx * Ry * Rz == R

@info "Setting up distributed nonhydrostatic model with N=($Nx, $Ny, $Nz) grid points and ranks=($Rx, $Ry, $Rz) on rank $local_rank..."

topo = (Periodic, Periodic, Periodic)
arch = Distributed(CPU(), topology=topo, ranks=(Rx, Ry, Rz), communicator=MPI.COMM_WORLD)
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
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            boundary_conditions = (u=u_boundary_conditions, b=b_boundary_conditions))
@show model

@inline Ξ(z) = randn() * exp(z / 4)

@inline stratification(z) = z < - initial_mixed_layer_depth ? N² * z : N² * (-initial_mixed_layer_depth)

@inline bᵢ(x, y, z) = stratification(z) + 1e-1 * Ξ(z) * N² * model.grid.Lz

u★ = sqrt(abs(τx))
@inline uᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
@inline wᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, b=bᵢ)

simulation = Simulation(model, Δt=45.0, stop_time=240hours)
@show simulation

conjure_time_step_wizard!(simulation, cfl=1.0, max_Δt=1minute)

output_interval = 1*hours

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
