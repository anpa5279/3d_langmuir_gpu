using Pkg
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BuoyancyFormulations: g_Earth
const Nx = 32        # number of points in each of x direction
const Ny = 32        # number of points in each of y direction
const Nz = 64        # number of points in the vertical direction
const Lx = 320    # (m) domain horizontal extents
const Ly = 320    # (m) domain horizontal extents
const Lz = 96    # (m) domain depth 
const N² = 5.3e-9    # s⁻², initial and bottom buoyancy gradient
const initial_mixed_layer_depth = 30.0 # m 
const Q = 1e11     # W m⁻², surface heat flux. cooling is positive
const cᴾ = 4200.0    # J kg⁻¹ K⁻¹, specific heat capacity of seawater
const ρₒ = 1026.0    # kg m⁻³, average density at the surface of the world ocean
const ρ_calcite = 2710.0 # kg m⁻³, dummy density of CaCO3
const dTdz = 0.01  # K m⁻¹, temperature gradient
const T0 = 25.0    # C, temperature at the surface  
const S₀ = 35.0    # ppt, salinity 
const β = 2.0e-4     # 1/K, thermal expansion coefficient
const u₁₀ = 5.75   # (m s⁻¹) wind speed at 10 meters above the ocean
const La_t = 0.3  # Langmuir turbulence number
#referring to files with desiraed functions
grid = RectilinearGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)) #arch
#stokes drift
include("stokes.jl")
dusdz = Field{Nothing, Nothing, Center}(grid)
z_d = collect(-Lz + grid.z.Δᵃᵃᶜ/2 : grid.z.Δᵃᵃᶜ : -grid.z.Δᵃᵃᶜ/2)
dusdz_1d = dstokes_dz.(z_d, u₁₀)
set!(dusdz, reshape(dusdz_1d, 1, 1, :))
@show dusdz
#BCs
u_f = La_t^2 * (stokes_velocity(-grid.z.Δᵃᵃᶜ/2, u₁₀)[1])
τx = -(u_f^2)
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx)) 
#@inline surface_heat_flux(x, y, t, p) = p.q / ( p.c *  p.ρ *  p.lx *  p.ly)/sqrt(2*pi* (p.σ^2)) * exp(-((x -  p.lx/2)^2 + (y -  p.ly/2)^2) / (2 * (p.σ)^2))
coriolis = FPlane(f=1e-4) # s⁻¹
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = β), constant_salinity = S₀)
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0.0), bottom = GradientBoundaryCondition(0.0))#surface_heat_flux, parameters = (q = Q, c = cᴾ, ρ = ρₒ, lx = Lx, ly = Ly, σ = 10.0)
#defining model
model = NonhydrostaticModel(; grid, buoyancy, coriolis,
                            advection = WENO(),
                            tracers = (:T,),
                            timestepper = :RungeKutta3,
                            closure = Smagorinsky(), 
                            stokes_drift = UniformStokesDrift(∂z_uˢ=dusdz),
                            boundary_conditions = (u=u_bcs, T=T_bcs))
@show model
# ICs
r_xy(a) = randn(Xoshiro(1234), 3 * Nx)[Int(1 + round((Nx) * a/(Lx + grid.Δxᶜᵃᵃ)))]
r_z(z) = randn(Xoshiro(1234), Nz +1)[Int(1 + round((Nz) * z/(-Lz)))] * exp(z/4)
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+dTdz * model.grid.Lz * 1e-6 * r_z(z) * r_xy(y) * r_xy(x + Lx)
uᵢ(x, y, z) = u_f * 1e-1  * r_z(z) * r_xy(y) * r_xy(x + Lx)
vᵢ(x, y, z) = -u_f * 1e-1  * r_z(z) * r_xy(y) * r_xy(x + Lx)
set!(model, u=uᵢ, v=vᵢ, T=Tᵢ)
day = 24hours
simulation = Simulation(model, Δt=30, stop_time = 0.5*day) #stop_time = 96hours,
@show simulation
# outputs and running
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
simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))
conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, max_Δt=30seconds)
#output files
function save_IC!(file, model)
    file["IC/friction_velocity"] = u_f
    file["IC/stokes_velocity"] = stokes_velocity(-grid.z.Δᵃᵃᶜ/2, u₁₀)[1]
    file["IC/wind_speed"] = u₁₀
    return nothing
end
output_interval = 0.25hours
u, v, w = model.velocities
T = model.tracers.T
simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T),
                                                    schedule = TimeInterval(output_interval),
                                                    filename = "localoutputs/T-NBP_fields.jld2", #$(rank)
                                                    overwrite_existing = true,
                                                    init = save_IC!)
W = Average(w, dims=(1, 2))
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
T = Average(T, dims=(1, 2))
                                                      
simulation.output_writers[:averages] = JLD2Writer(model, (; U, V, W, T),
                                                    schedule = AveragedTimeInterval(output_interval, window=output_interval),
                                                    filename = "localoutputs/T-NBP_averages.jld2",
                                                    overwrite_existing = true)
run!(simulation)#; pickup = true)