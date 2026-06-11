using ThreadPinning
using MPI
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
size = MPI.Comm_size(MPI.COMM_WORLD)
nthreads = Threads.nthreads()
mpi_pinthreads(:numa)
using Pkg
using JLD2
using Statistics
using Printf
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minute, minutes, hours, seconds

Lx = Ly = 128           # (m) domain horizontal extents
Nx = Ny = 64*2^2 #ensure it is only powers of 2 (maybe 3)

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

arch = Distributed(CPU())
# defining grid
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = (-Lz, 0))
# Save grid metadata to a separate file (rank 0 only)
if rank == 0
    jldopen("grid_info.jld2", "w") do file
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
        file["grid/arch"]   = string(arch)
        file["grid/Nranks"] = MPI.Comm_size(MPI.COMM_WORLD)
    end
end
# buoyancy
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = alpha))

# BCs
@inline function sflux(x, y, t) 
    if x<=rp && y<=rp
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
Tᵢ(x, y, z) = z > - MLD ? T0 : T0 + dTdz * (z + MLD)

set!(model, u=0.0, v=0.0, T=Tᵢ, S=0.0)

# defining simulation
simulation = Simulation(model, Δt=min_step, stop_time = 4hours, minimum_relative_step = 0.01) 
if rank == 0 
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
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))
    @show simulation
end
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
                                                    filename = "fields.jld2",
                                                    overwrite_existing = true)

v_avg_oc = Average(v, dims=(1, 2))
w_avg_oc = Average(w, dims=(1, 2))
T_avg_oc = Average(T, dims=(1, 2))
u_avg_oc = Average(u, dims=(1, 2))
simulation.output_writers[:xy_avg_oc] = JLD2Writer(model, (; u_avg_oc, v_avg_oc, w_avg_oc, T_avg_oc), # test within if statement and outside of
                                                    with_halos=false,
                                                    array_type = Array{Float64},
                                                    schedule = TimeInterval(output_interval/100),
                                                    filename = "xy_avg_oc.jld2",
                                                    overwrite_existing = true)
# ---- Global average storage (rank 0 only needs the final result) ----
# These live outside callbacks so they persist between calls
u_avg_global = zeros(Float64, 1, 1, Nz)   # shape matches z-column
v_avg_global = zeros(Float64, 1, 1, Nz)
w_avg_global = zeros(Float64, 1, 1, Nz)
S_avg_global = zeros(Float64, 1, 1, Nz)
T_avg_global = zeros(Float64, 1, 1, Nz)

function global_avg!(simulation)
    u, v, w = simulation.model.velocities
    T = simulation.model.tracers.T
    S = simulation.model.tracers.S

    # Local horizontal mean — result is (1, 1, Nz)
    u_avg_local = mean(interior(u), dims=(1, 2))
    v_avg_local = mean(interior(v), dims=(1, 2))
    w_avg_local = mean(interior(w), dims=(1, 2))
    T_avg_local = mean(interior(T), dims=(1, 2))
    S_avg_local = mean(interior(S), dims=(1, 2))

    # MPI_Allreduce (or Reduce to rank 0) sums across all ranks,
    # then divide by number of ranks to get the true mean
    MPI.Allreduce!(u_avg_local, u_avg_global, MPI.SUM, MPI.COMM_WORLD)
    MPI.Allreduce!(v_avg_local, v_avg_global, MPI.SUM, MPI.COMM_WORLD)
    MPI.Allreduce!(w_avg_local, w_avg_global, MPI.SUM, MPI.COMM_WORLD)
    MPI.Allreduce!(T_avg_local, T_avg_global, MPI.SUM, MPI.COMM_WORLD)
    MPI.Allreduce!(S_avg_local, S_avg_global, MPI.SUM, MPI.COMM_WORLD)

    u_avg_global ./= size   # 'size' = MPI.Comm_size(MPI.COMM_WORLD)
    v_avg_global ./= size
    w_avg_global ./= size
    T_avg_global ./= size
    S_avg_global ./= size

    return nothing
end
simulation.callbacks[:global_avg] = Callback(global_avg!, TimeInterval(output_interval / 100))

function global_fluc_sq_avg!(simulation)
    u, v, w = simulation.model.velocities
    T = simulation.model.tracers.T
    S = simulation.model.tracers.S

    # w_mean_global must already be up to date from a prior callback
    _u_sq_local .= mean((interior(u) .- u_mean_global) .^ 2, dims=(1, 2))
    _v_sq_local .= mean((interior(v) .- v_mean_global) .^ 2, dims=(1, 2))
    _w_sq_local .= mean((interior(w) .- w_mean_global) .^ 2, dims=(1, 2))
    _T_sq_local .= mean((interior(T) .- T_mean_global) .^ 2, dims=(1, 2))
    _S_sq_local .= mean((interior(S) .- S_mean_global) .^ 2, dims=(1, 2))

    MPI.Allreduce!(_u_sq_local, u_fluc_sq_avg_global, MPI.SUM, MPI.COMM_WORLD)
    MPI.Allreduce!(_v_sq_local, v_fluc_sq_avg_global, MPI.SUM, MPI.COMM_WORLD)
    MPI.Allreduce!(_w_sq_local, w_fluc_sq_avg_global, MPI.SUM, MPI.COMM_WORLD)
    MPI.Allreduce!(_T_sq_local, T_fluc_sq_avg_global, MPI.SUM, MPI.COMM_WORLD)
    MPI.Allreduce!(_S_sq_local, S_fluc_sq_avg_global, MPI.SUM, MPI.COMM_WORLD)

    u_fluc_sq_avg_global ./= size
    v_fluc_sq_avg_global ./= size
    w_fluc_sq_avg_global ./= size
    T_fluc_sq_avg_global ./= size
    S_fluc_sq_avg_global ./= size

    return nothing
end

simulation.callbacks[:global_fluc_sq_avg]  = Callback(global_fluc_sq_avg!,  TimeInterval(output_interval/100))

if rank == 0
    # Wrap the global arrays as FieldTimeSeries-compatible outputs.
    # The simplest approach: write them as plain arrays via a Dict.
    simulation.output_writers[:xy_avg] = JLD2Writer(
        model,
        Dict("u_avg" => model -> u_avg_global,
             "v_avg" => model -> v_avg_global,
             "w_avg" => model -> w_avg_global,
             "T_avg" => model -> T_avg_global,
             "S_avg" => model -> S_avg_global, 
             "u_fluc_sq_avg" => model -> u_fluc_sq_avg_global,
             "v_fluc_sq_avg" => model -> v_fluc_sq_avg_global,
             "w_fluc_sq_avg" => model -> w_fluc_sq_avg_global,
             "T_fluc_sq_avg" => model -> T_fluc_sq_avg_global,
             "S_fluc_sq_avg" => model -> S_fluc_sq_avg_global),
        schedule = TimeInterval(output_interval / 100),
        filename = "xy_avg.jld2",
        overwrite_existing = true
    )
end
run!(simulation)