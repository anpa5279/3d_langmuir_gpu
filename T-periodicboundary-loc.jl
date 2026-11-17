using Pkg
using JLD2
using Printf
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minutes, hours, seconds

const Lx = Ly = 256           # (m) domain horizontal extents
const Nx = Ny = 128 #ensure it is only powers of 2 (maybe 3)

const Lz = 128             # (m) domain depth 
const Nz = 64
const MLD = 40.0          # m, mixed layer depth
const dTdz = 0.01       # K m⁻¹, temperature gradient
const alpha = 2.0e-4      # 1/K, thermal expansion coefficient
const rp = 4.0           # m, radius of surface buoyancy flux
const T0 = 25.0 + 273.15 # K, temperature at the surface
const min_step = 0.01
const wp = -0.001 # m/s, vertical velocity for surface buoyancy flux

const Ln = rp # m, Morton's length scale 

dir = "localoutputs/time output field test/"
# defining grid
grid = RectilinearGrid(; size=(Nx, Ny, Nz), x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = (-Lz, 0))

# Save grid metadata to a separate file (rank 0 only)
jldopen("$(dir)grid_info.jld2", "w") do file
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

# calculating mass relative to morton length scale
const g = buoyancy.gravitational_acceleration
const beta = buoyancy.equation_of_state.haline_contraction
const N = sqrt(dTdz * alpha * g)
const F0 = (Ln * N^(3/4))^4
const Sj = F0 / (g * beta * abs(wp) * pi * rp^2)

# BCs
u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),
                                bottom = GradientBoundaryCondition(dTdz))

@inline gauss(x, y) = exp(-((x^2 + y^2)/(2*rp^2)))
@inline w_surface(x, y, t)=wp*gauss(x, y)

w_bottom = wp * 2*pi*rp^2 / (Lx*Ly)

w_bcs = FieldBoundaryConditions(top = OpenBoundaryCondition(w_surface;  scheme = PerturbationAdvection(; inflow_timescale = 0.0, outflow_timescale = 0.0)), 
                                bottom = OpenBoundaryCondition(w_bottom;  scheme = PerturbationAdvection(; inflow_timescale = 0.0, outflow_timescale = 0.0)))

@inline S_surface(x, y, t)=Sj*gauss(x, y)

S_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(S_surface), 
                                bottom = GradientBoundaryCondition(0.0))

## defining model
model = NonhydrostaticModel(grid;
                            buoyancy, 
                            advection = WENO(order = 9),
                            tracers = (:T, :S,),
                            timestepper = :RungeKutta3,
                            boundary_conditions = (u = u_bcs, v = v_bcs, w=w_bcs, S=S_bcs, T=T_bcs),
                            )
@show model
## ICs
Tᵢ(x, y, z) = z > - MLD ? T0 : T0 + dTdz * (z + MLD)

set!(model, u=0.0, v=0.0, T=Tᵢ, S=0.0)

# defining simulation
simulation = Simulation(model, Δt=0.5/0.1*Lz/Nz, stop_time = 4hours, minimum_relative_step = 0.01) 
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

## updating cfl every time step
conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, diffusive_cfl = 1.0, min_change = 0.1, max_change = 1.5, min_Δt = min_step, max_Δt=30seconds) #ensures cfl is updated ever iteration
## output files
output_interval = 0.01hours
u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
get_Δt = let sim = simulation
    model -> sim.Δt
end

simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T, S),
                                                with_halos=false,
                                                array_type = Array{Float64},
                                                schedule = TimeInterval(output_interval),
                                                dir = dir,
                                                filename = "fields.jld2",
                                                overwrite_existing = true)
simulation.output_writers[:dt] = JLD2Writer(model, (; Δt = get_Δt),
                                        schedule = TimeInterval(output_interval),
                                        dir = dir,
                                        filename = "dt.jld2",
                                        overwrite_existing = true
                                    )
simulation.output_writers[:centerline] = JLD2Writer(model, (; u, v, w, T, S), # test within if statement and outside of 
                                                    indices = (Int(grid.Nx/2):Int(grid.Nx/2+1), Int(Ny/2):Int(Ny/2+1), :),
                                                    array_type = Array{Float64},
                                                    schedule = TimeInterval(output_interval/20),
                                                    dir = dir,
                                                    filename = "centerline.jld2",
                                                    overwrite_existing = true)

v_avg = Average(v, dims=(1, 2))
w_avg = Average(w, dims=(1, 2))
T_avg = Average(T, dims=(1, 2))
S_avg = Average(S, dims=(1, 2))
u_avg = Average(u, dims=(1, 2))
simulation.output_writers[:xy_avg] = JLD2Writer(model, (; u_avg, v_avg, w_avg, T_avg, S_avg), # test within if statement and outside of
                                                with_halos=false,
                                                array_type = Array{Float64},
                                                schedule = TimeInterval(output_interval/20),
                                                dir = dir,
                                                filename = "xy_avg.jld2",
                                                overwrite_existing = true)

run!(simulation)