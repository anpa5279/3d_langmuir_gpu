using Pkg
using JLD2
using Statistics
using Printf
using SpecialFunctions
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minute, minutes, hours, seconds

Lx = Ly = 128           # (m) domain horizontal extents
Nx = Ny = 128 #ensure it is only powers of 2 (maybe 3)

Lz = 128             # (m) domain depth 
Nz = 256
MLD = 60.0          # m, mixed layer depth
dTdz = 0.01       # K m⁻¹, temperature gradient
alpha = 2.0e-4      # 1/K, thermal expansion coefficient
rp = 4.0           # m, radius of surface buoyancy flux
T0 = 25.0           # C, temperature at the surface
min_step = 0.01
wp = -0.001 # m/s, vertical velocity for surface buoyancy flux
Sj = 0.1 # g/kg, tracer mass 

# defining grid
grid = RectilinearGrid(; size=(Nx, Ny, Nz), x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = (-Lz, 0))
# Save grid metadata to a separate file (rank 0 only)
jldopen("localoutputs/grid_info.jld2", "w") do file
    file["grid/x"]      = grid.xᶜᵃᵃ
    file["grid/y"]      = grid.yᵃᶜᵃ
    file["grid/z"]      = grid.z.cᵃᵃᶜ
    file["grid/Δx"]     = grid.Δxᶜᵃᵃ
    file["grid/Δy"]     = grid.Δyᵃᶜᵃ
    file["grid/Δz"]     = grid.z.Δᵃᵃᶜ
    file["grid/Nx"]     = Nx
    file["grid/Ny"]     = Ny
    file["grid/Nz"]     = Nz
    file["grid/Lx"]     = Lx
    file["grid/Ly"]     = Ly
    file["grid/Lz"]     = Lz
end

# buoyancy
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = alpha))

# BCs
@inline function sflux(x, y, t) 
    if abs(x)<=rp && abs(y)<=rp
        return wp*Sj
    else
        return 0.0
    end
end
u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),
                                bottom = GradientBoundaryCondition(dTdz))
S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(sflux), 
                                bottom = GradientBoundaryCondition(0.0))
## defining model
model = NonhydrostaticModel(grid;
                            buoyancy, 
                            advection = WENO(; minimum_buffer_upwind_order = 1),
                            tracers = (:T, :S,),
                            timestepper = :RungeKutta3,
                            boundary_conditions = (u = u_bcs, v = v_bcs, S=S_bcs, T=T_bcs),
                            )
@show model
## ICs
a = dTdz*sqrt(pi)/2
T1 = T0 - a
Tᵢ(x, y, z) = z > - MLD ? a * erf(z + MLD) + T0 - a : T1 + dTdz * (z + MLD)

set!(model, u=0.0, v=0.0, T=Tᵢ, S=0.0)

# defining simulation
simulation = Simulation(model, Δt=min_step, stop_time = 4hours)#, minimum_relative_step = 0.01) 
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
simulation.callbacks[:progress] = Callback(progress, IterationInterval(500))
@show simulation

## updating cfl every time step
conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, diffusive_cfl = 1.0, min_Δt = min_step, max_Δt=30seconds) #ensrues cfl is updated ever iteration
## output files
output_interval = 0.2hours
u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S

simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T, S),
                                                with_halos=false,
                                                array_type = Array{Float64},
                                                schedule = TimeInterval(output_interval),
                                                dir = "localoutputs",
                                                filename = "fields.jld2",
                                                overwrite_existing = true)

simulation.output_writers[:centerline] = JLD2Writer(model, (; u, v, w, T, S), # test within if statement and outside of 
                                                indices = (Int(grid.Nx):Int(grid.Nx+1), Int(Ny/2):Int(Ny/2+1), :),
                                                array_type = Array{Float64},
                                                schedule = TimeInterval(output_interval/100),
                                                dir = "localoutputs",
                                                filename = "centerline.jld2",
                                                overwrite_existing = true)

run!(simulation)