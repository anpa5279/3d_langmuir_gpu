using ThreadPinning
using MPI
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
size = MPI.Comm_size(MPI.COMM_WORLD)
nthreads = Threads.nthreads()
mpi_pinthreads(:numa)
using Pkg
using JLD2
using HDF5
using Statistics
using Printf
using Oceananigans
import Oceananigans.AbstractOperations: Average
using Oceananigans.Units: minute, minutes, hours, seconds

rank = 0
size = 2
Lx = Ly = 128           # (m) domain horizontal extents
Nx = Ny = 64*2^2 #ensure it is only powers of 2 (maybe 3)
idx = Int.(range(1, Nx, step = 1))
Nx_loc = Nx / size

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
    end
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
## closure 
visc = 1e-6
closure = ScalarDiffusivity(ν=visc, κ=visc)
## defining model
model = NonhydrostaticModel(grid;
                            buoyancy, 
                            advection = WENO(; minimum_buffer_upwind_order = 1),
                            tracers = (:T, :S,),
                            timestepper = :RungeKutta3,
                            boundary_conditions = (u = u_bcs, v = v_bcs, S=S_bcs, T=T_bcs),
                            closure = closure,
                            )
## ICs
#T_nt = load(file, "T")
#u_nt = load(file, "u")
#v_nt = load(file, "v")
#w_nt = load(file, "w")
#S_nt = load(file, "S")

#set!(model, u=u_nt, v=v_nt, w = w_nt, T=T_nt, S=S_nt)
file = "last_time_step.jld2"
jldopen(file, "r") do file
    T_nt = file["T"][idx[Int(rank*Nx_loc+1):Int((rank+1)*Nx_loc)], :, :]
    u_nt = file["u"][idx[Int(rank*Nx_loc+1):Int((rank+1)*Nx_loc)], :, :]
    v_nt = file["v"][idx[Int(rank*Nx_loc+1):Int((rank+1)*Nx_loc)], :, :]
    w_nt = file["w"][idx[Int(rank*Nx_loc+1):Int((rank+1)*Nx_loc)], :, :]
    S_nt = file["S"][idx[Int(rank*Nx_loc+1):Int((rank+1)*Nx_loc)], :, :]
    set!(model, u=u_nt, v=v_nt, w = w_nt, T=T_nt, S=S_nt)
end

@show model
# defining simulation
simulation = Simulation(model, Δt=min_step, stop_time = 4hours, minimum_relative_step = 0.01) 

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
                                                filename = "fields_pickup.jld2",
                                                overwrite_existing = true)

v_avg = Average(v, dims=(1, 2))
w_avg = Average(w, dims=(1, 2))
T_avg = Average(T, dims=(1, 2))
S_avg = Average(S, dims=(1, 2))
u_avg = Average(u, dims=(1, 2))
simulation.output_writers[:xy_avg] = JLD2Writer(model, (; u_avg, v_avg, w_avg, T_avg, S_avg), # test within if statement and outside of
                                                with_halos=false,
                                                array_type = Array{Float64},
                                                schedule = TimeInterval(output_interval/100),
                                                filename = "xy_avg_pickup.jld2",
                                                overwrite_existing = true)

run!(simulation)