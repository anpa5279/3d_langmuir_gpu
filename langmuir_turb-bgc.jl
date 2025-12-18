using Pkg
using MPI
using CUDA
@show MPI.has_cuda()
@show CUDA.has_cuda()
MPI.Init() # Initialize MPI
using Random
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minute, minutes, hours, seconds
using Printf
using Oceananigans.DistributedComputations
using Oceananigans.TurbulenceClosures: AnisotropicMinimumDissipation, Smagorinsky
#using Oceananigans.Diagnostics: NaNChecker
Pkg.develop(path="/glade/work/apauls/personal-oceananigans/OceanBioME.jl-main/")
using OceanBioME: CarbonateChemistry, carbonate_thermo_kernel
using OceanBioME: ContinuousBiogeochemistry
using Oceananigans.AbstractOperations: KernelFunctionOperation

const Nx = 2        # number of points in each of x direction
const Ny = 2        # number of points in each of y direction
const Nz = 2        # number of points in the vertical direction
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
const S0 = 35.0    # ppt, salinity 
const β = 2.0e-4     # 1/K, thermal expansion coefficient
const u₁₀ = 5.75   # (m s⁻¹) wind speed at 10 meters above the ocean
const La_t = 0.3  # Langmuir turbulence number

# Automatically distribute among available processors
Nranks = MPI.Comm_size(MPI.COMM_WORLD)
arch = Nranks > 1 ? Distributed(GPU()) : GPU()

# Determine rank safely depending on architecture
rank = arch isa Distributed ? arch.local_rank : 0
Nranks = arch isa Distributed ? MPI.Comm_size(arch.communicator) : 1

# defining domain and grid
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
@show grid  
"""
# other forcing
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = β), constant_salinity = S0)
coriolis = FPlane(f=1e-4) # s⁻¹

# stokes drift
g = buoyancy.gravitational_acceleration

amplitude = 0.8 # m
wavelength = 60  # m
wavenumber = 2π / wavelength # m⁻¹
frequency = sqrt(g * wavenumber) # s⁻¹

const vertical_scale = wavelength / 4π

# Stokes drift velocity at the surface
const us = amplitude^2 * wavenumber * frequency # m s⁻¹
uˢ(z) = us * exp(z / vertical_scale)
∂z_uˢ(z, t) = 1 / vertical_scale * us * exp(z / vertical_scale)

# BCs
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q / (cᴾ * ρₒ * Lx * Ly)),
                                bottom = GradientBoundaryCondition(dTdz))

u_f = La_t^2 * us
const τx = -(u_f^2)# m² s⁻², surface kinematic momentum flux
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx), 
                                bottom = GradientBoundaryCondition(0.0))

v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), #ValueBoundaryCondition(0.0), #
                                bottom = GradientBoundaryCondition(0.0))
# ICs
"""
r_z(z) = z > - initial_mixed_layer_depth ? randn(Xoshiro()) : 0.0 
ampv = 1.0e-3 # m s⁻¹
ue(x, y, z) = ampv * r_z(z)
uᵢ(x, y, z) = -ue(x, y, z) #+ uˢ(z)
vᵢ(x, y, z) = ue(x, y, z)
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? (T0 + dTdz * model.grid.Lz * ampv * r_z(z)) : T0 + dTdz * (z + initial_mixed_layer_depth) 

# BGC model
biogeochemistry = CarbonateChemistry(; grid, scale_negatives=true)

#  defining model
model = NonhydrostaticModel(; grid, #coriolis,
                            #advection = WENO(order=9), 
                            biogeochemistry = biogeochemistry,
                            auxiliary_fields = (K1= CenterField(grid), 
                                                K2= CenterField(grid), 
                                                Kw= CenterField(grid), 
                                                Kb= CenterField(grid), 
                                                a1= CenterField(grid), 
                                                a2= CenterField(grid), 
                                                a6= CenterField(grid), 
                                                a7= CenterField(grid), 
                                                b1= CenterField(grid), 
                                                b2= CenterField(grid), 
                                                b3= CenterField(grid), 
                                                b4= CenterField(grid), 
                                                b5= CenterField(grid), 
                                                b6= CenterField(grid), 
                                                b7= CenterField(grid), 
                                                H= CenterField(grid)),
                            timestepper = :RungeKutta3,
                            tracers = (:CO2, :HCO3, :CO3, :OH, :BOH3, :BOH4, :T),
                            #buoyancy = buoyancy,
                            #closure = AnisotropicMinimumDissipation(), #
                            #stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            #boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs) 
                            )
@show model
# ICs
perturb = 1e3
set!(model, w=0.0, u=uᵢ, v=vᵢ, T=Tᵢ, BOH3 = 2.97e2, BOH4 = 1.19e2, CO2 = 7.57e0 * perturb, CO3 = 3.15e2, HCO3 = 1.67e3, OH = 9.6e0) 
@show "ICs set"

simulation = Simulation(model, Δt=3e-7, stop_time=5.0)
@show simulation
function progress(simulation)
    u, v, w = simulation.model.velocities

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n
    CO2 = %.1e, CO3 = %.1e, HCO3 = %.1e, oh = %.1e, BOH3 = %.1e, BOH4 = %.1e",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(simulation.run_wall_time), 
                   maximum(simulation.model.tracers.CO2),
                   maximum(simulation.model.tracers.CO3),
                   maximum(simulation.model.tracers.HCO3),
                   maximum(simulation.model.tracers.OH),
                   maximum(simulation.model.tracers.BOH3),
                   maximum(simulation.model.tracers.BOH4))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(5000))
simulation.callbacks[:cc_updates] = Callback(carbonate_thermo_kernel, IterationInterval(1), callsite=UpdateStateCallsite())

output_interval =  0.1*seconds

u, v, w = model.velocities
BOH3 = model.tracers.BOH3
BOH4 = model.tracers.BOH4
CO2 = model.tracers.CO2
CO3 = model.tracers.CO3
HCO3 = model.tracers.HCO3
OH = model.tracers.OH
T = model.tracers.T

simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T, BOH3, BOH4, CO2, CO3, HCO3, OH),
                                                    schedule = TimeInterval(output_interval),
                                                    filename = "vel_tracer_fields.jld2",
                                                    overwrite_existing = true,
                                                    with_halos = false,
                                                    array_type = Array{Float64}
                                                    )
                                                      
simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(5000), prefix="model_checkpoint")

run!(simulation)#; pickup = true)