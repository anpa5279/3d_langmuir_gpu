using Pkg
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans: defaults #using Oceananigans.BuoyancyFormulations: g_Earth
using Oceananigans.TurbulenceClosures: Smagorinsky
const Nx = 8        # number of points in each of x direction
const Ny = 8        # number of points in each of y direction
const Nz = 8        # number of points in the vertical direction
const Lx = 320    # (m) domain horizontal extents
const Ly = 320    # (m) domain horizontal extents
const Lz = 96    # (m) domain depth 
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

grid = RectilinearGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)) #arch

#stokes drift
g_Earth = defaults.gravitational_acceleration
include("stokes.jl")
dusdz = Field{Nothing, Nothing, Center}(grid)
z_d = collect(-Lz + grid.z.Δᵃᵃᶜ/2 : grid.z.Δᵃᵃᶜ : -grid.z.Δᵃᵃᶜ/2)
dusdz_1d = dstokes_dz.(z_d, u₁₀)
set!(dusdz, reshape(dusdz_1d, 1, 1, :))
@show dusdz

#BCs
us = stokes_velocity(z_d[end], u₁₀)
u_f = La_t^2 * us
τx = -(u_f^2)# m² s⁻², surface kinematic momentum flux
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx), bottom = GradientBoundaryCondition(0.0)) 
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), bottom = GradientBoundaryCondition(0.0))

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = β), constant_salinity = S0)

T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),#FluxBoundaryCondition(Q / (cᴾ * ρₒ * Lx * Ly)),
                                bottom = GradientBoundaryCondition(0.0))
#coriolis = FPlane(f=1e-4) # s⁻¹

model = NonhydrostaticModel(; grid, buoyancy, #coriolis,
                            advection = WENO(),
                            tracers = (:T,),
                            timestepper = :RungeKutta3,
                            closure = Smagorinsky(coefficient=0.1),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=dusdz),
                            boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs, ))
@show model

# ICs
r_z(z) = randn(Xoshiro()) * exp(z/4)
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+ dTdz * model.grid.Lz * 1e-6 * r_z(z) 
uᵢ(x, y, z) = -u_f * r_z(z) 
vᵢ(x, y, z) = -uᵢ(x, y, z)
set!(model, u=uᵢ, w=0.0, T=Tᵢ)#v=vᵢ, 

day = 24hours
simulation = Simulation(model, Δt=30, stop_time = 10*day) #stop_time = 96hours,
@show simulation

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

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))

conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, max_Δt=30seconds)

#output files
function save_IC!(file, model)
    file["IC/friction_velocity"] = u_f
    file["IC/stokes_velocity"] = us
    file["IC/wind_speed"] = u₁₀
    return nothing
end

output_interval = 2hours

u, v, w = model.velocities
T = model.tracers.T
W = Average(w, dims=(1, 2))
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
T_avg = Average(T, dims=(1, 2))

simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T),
                                                    schedule = TimeInterval(output_interval),
                                                    filename = "langmuir_turbulence_fields.jld2", #$(rank)
                                                    overwrite_existing = true,
                                                    init = save_IC!)
                                                      
simulation.output_writers[:averages] = JLD2Writer(model, (; U, V, W, T_avg),
                                                    schedule = AveragedTimeInterval(output_interval, window=output_interval),
                                                    filename = "langmuir_turbulence_averages.jld2",
                                                    overwrite_existing = true)
#simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(30000), prefix="model_checkpoint")

run!(simulation)#; pickup = true)