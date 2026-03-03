using Pkg
using Random
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
using Printf
using Oceananigans.TurbulenceClosures: AnisotropicMinimumDissipation, Smagorinsky
using Oceananigans.BoundaryConditions: fill_halo_regions!
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
S0 = 35.0    # ppt, salinity 
alpha = 2.0e-4     # 1/K, thermal expansion coefficient
u₁₀ = 5.75   # (m s⁻¹) wind speed at 10 meters above the ocean
La_t = 0.3084  # Langmuir turbulence number

grid = RectilinearGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
# other forcing
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = alpha), constant_salinity = S0)

# BCs
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q / (cᴾ * ρₒ)),
                                bottom = GradientBoundaryCondition(dTdz))

model = NonhydrostaticModel(grid; 
                            timestepper = :RungeKutta3,
                            tracers = :T,
                            buoyancy = buoyancy,
                            boundary_conditions = (T=T_bcs,)
                            )
@show model
# ICs
r(x, y, z) = randn(Xoshiro(1234), (Nx + Ny +Nz+3))[Int(1 + round(Nx*x/Lx+Ny*y/Ly-Nz*z/Lz))] * exp(z / 4) 
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+dTdz * model.grid.Lz * 1e-6 * r(x, y, z)
uᵢ(x, y, z) = u_f * r(x, y, z)
set!(model, w=0.0, u = uᵢ, T=Tᵢ) 
@show "ICs set"

simulation = Simulation(model, Δt=30.0, stop_time=8hours)
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

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, max_Δt=30.0)

#output files
function save_IC!(file, model)
    file["IC/friction_velocity"] = u_f
    file["IC/stokes_velocity"] = stokes_velocity(-grid.z.Δᵃᵃᶜ/2, u₁₀)[1]
    return nothing
end
path = "localoutputs/buoyancy testing/celsius/withrand"
output_interval = 0.1*hours

u, v, w = model.velocities
T = model.tracers.T
W = Average(w, dims=(1, 2))
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))

simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T),
                                                    dir = path,  with_halos=false,
                                                    array_type = Array{Float64},
                                                    schedule = TimeInterval(output_interval),
                                                    filename = "langmuir_turbulence_fields.jld2", #$(rank)
                                                    overwrite_existing = true,
                                                    init = save_IC!)
                                                      
T = Average(T, dims=(1, 2))
simulation.output_writers[:averages] = JLD2Writer(model, (; U, V, W, T),
                                                    dir = path,  with_halos=false,
                                                    array_type = Array{Float64},
                                                    schedule = AveragedTimeInterval(output_interval, window=output_interval),
                                                    filename = "langmuir_turbulence_averages.jld2",
                                                    overwrite_existing = true)
run!(simulation)#; pickup = true)
