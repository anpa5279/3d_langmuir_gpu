using Pkg
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BoundaryConditions: fill_halo_regions!, OpenBoundaryCondition
using Oceananigans.Models: BoundaryAdjacentMean
using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.TurbulenceClosures: Smagorinsky
## simulation parameters
Nx = 32        # number of points in each of x direction
Ny = 32        # number of points in each of y direction
Nz = 64        # number of points in the vertical direction
Lx = 320    # (m) domain horizontal extents
Ly = 320    # (m) domain horizontal extents
Lz = 96    # (m) domain depth 
initial_mixed_layer_depth = 30.0 # m 
Q = 5.0     # W m⁻², surface heat flux. cooling is positive
cᴾ = 4200.0    # J kg⁻¹ K⁻¹, specific heat capacity of seawater
ρₒ = 1026.0    # kg m⁻³, average density at the surface of the world ocean
dTdz = 0.01  # K m⁻¹, temperature gradient
T0 = 25.0    # C, temperature at the surface  
S₀ = 35.0    # ppt, salinity 
β = 2.0e-4     # 1/K, thermal expansion coefficient
u₁₀ = 5.75   # (m s⁻¹) wind speed at 10 meters above the ocean
La_t = 0.3  # Langmuir turbulence number

const ρ_calcite = 2710.0 # kg m⁻³, dummy density of CaCO3
const molar_calcite = 100.09/1000.0 # kg/mol, molar mass of CaCO3
mass = 1000

## referring to files with desiraed functions
grid = RectilinearGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)) #arch
## stokes drift
include("stokes.jl")
u_f = La_t^2 * (stokes_velocity(-grid.z.Δᵃᵃᶜ/2, u₁₀)[1])
τx = -(u_f^2)
u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), #FluxBoundaryCondition(τx), 
                                bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), #FluxBoundaryCondition(Q/(ρₒ*cᴾ)),
                                bottom = GradientBoundaryCondition(dTdz))
@inline function CaCO3_t(x, y, t) 
    if (t <= 6hours)
        σ = 10.0 # m
        c0 = mass/(molar_calcite*(Lx/Nx)*(Ly/Ny)*(Lz/Nz)) # mol/m3
        return c0/sqrt(2*pi* σ^2) * exp(-(x-Lx/2)^2 / (2 * σ^2)) * exp(-(y-Ly/2)^2 / (2 * σ^2)) 
    else
        return 0.0
    end
end
CaCO3_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(CaCO3_t), 
                                    bottom = GradientBoundaryCondition(0.0))

## defining forcing (coriolis, buoyancy, etc.)
coriolis = FPlane(f=1e-4) # s⁻¹
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = β), constant_salinity = S₀)
# defining forcing functions
include("NBP_forcing.jl")
w_NBP = Forcing(densescalar, discrete_form=true, parameters=(molar_masses = (molar_calcite,), densities = (ρ_calcite,), reference_density = ρₒ, thermal_expansion = β))

## defining model
model = NonhydrostaticModel(; grid, #coriolis, 
                            buoyancy, 
                            advection = WENO(),
                            tracers = (:T, :CaCO3),
                            timestepper = :RungeKutta3,
                            closure = Smagorinsky(), 
                            boundary_conditions = (u = u_bcs, v = v_bcs, T=T_bcs, CaCO3=CaCO3_bcs),
                            forcing = (w = w_NBP,))
@show model
## ICs
r(x, y, z) = (1+randn(Xoshiro())) * exp(z/4)
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+dTdz * model.grid.Lz * 1e-6 * r(x, y, z)
uᵢ(x, y, z) = u_f * r(x, y, z)
vᵢ(x, y, z) = -u_f * r(x, y, z)

σ = 10.0 # m
c0 = mass/(molar_calcite*(Lx/Nx)*(Ly/Ny)*(Lz/Nz)) # mol/m3
CaCO3ᵢ(x, y, z) = c0/sqrt(2*pi* σ^2) * exp(-z^2 / (2 * σ^2)) * exp(-(x-Lx/2)^2 / (2 * σ^2)) * exp(-(y-Ly/2)^2 / (2 * σ^2)) 

set!(model, u=uᵢ, v=vᵢ, T=Tᵢ, CaCO3=CaCO3ᵢ)

# defining simulation
simulation = Simulation(model, Δt=30, stop_time = 2.0*24hours) 
@show simulation
## progress function
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
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
## updating cfl every time step
conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, min_Δt = 1.0, max_Δt=30seconds) #ensrues cfl is updated ever iteration
## output files
function save_IC!(file, model)
    file["IC/friction_velocity"] = u_f
    file["IC/stokes_velocity"] = stokes_velocity(-grid.z.Δᵃᵃᶜ/2, u₁₀)[1]
    return nothing
end
output_interval = 0.25hours
path = "localoutputs/NBP no coriolis and no stokes and no fluxes/"
u, v, w = model.velocities
T = model.tracers.T
CaCO3 = model.tracers.CaCO3
P_static = model.pressures.pHY′
P_dynamic = model.pressures.pNHS
simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T, CaCO3, P_static, P_dynamic),
                                                    dir = path,  with_halos=false,
                                                    array_type = Array{Float64},
                                                    schedule = TimeInterval(output_interval),
                                                    filename = "NBP_fields.jld2", #$(rank)
                                                    overwrite_existing = true,
                                                    init = save_IC!)

# running the simulation
run!(simulation)#; pickup = true)
