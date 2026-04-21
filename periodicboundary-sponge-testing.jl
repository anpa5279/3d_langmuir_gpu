using Pkg
using Statistics
using Printf
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minute, minutes, hours, seconds
## simulation parameters
Nx = 48
Ny = 48
Nz = 48
Lx = 320            # (m) domain horizontal extents
Ly = 320            # (m) domain horizontal extents
Lz = 96             # (m) domain depth 
MLD = 30.0          # m, mixed layer depth
dTdz = 0.01       # K m⁻¹, temperature gradient
alpha = 2.0e-4      # 1/K, thermal expansion coefficient
rp = 5.0           # m, radius of surface buoyancy flux
T0 = 25.0           # C, temperature at the surface
min_step = 0.1
La_t = 0.3
wp = -0.001 # m/s, vertical velocity for surface buoyancy flux
Sj = 0.1 # g/kg, tracer mass 
import functions: progress, save_grid!, sflux
# defining grid
grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz), x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = (-Lz, 0))
@show grid
# buoyancy
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = alpha))

# BCs
u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),
                                bottom = GradientBoundaryCondition(dTdz))
S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(sflux), 
                                bottom = GradientBoundaryCondition(0.0))
# ICs
Tᵢ(x, y, z) = z > - MLD ? T0 : T0 + dTdz * (z + MLD)

# types of masks to test
mask_str = ["gaussian", "linear"]#["gaussian", "linear", "none"]
damp_rate = 1/60 # [1/s], relaxation rate for all masks
T_target(x, y, z, t) = Tᵢ(0, y, z)
# gaussian mask
gaus_mask_x1 = GaussianMask{:x}(center = Lx/2, width = Ly/32)
gaus_mask_x2 = GaussianMask{:x}(center = -Lx/2, width = Ly/32)
sponge_x_vel1 = Relaxation(; rate = damp_rate, mask = gaus_mask_x1)
sponge_x_T1 = Relaxation(; rate = damp_rate, mask = gaus_mask_x1, target = T_target)
sponge_x_vel2 = Relaxation(; rate = damp_rate, mask = gaus_mask_x2)
sponge_x_T2 = Relaxation(; rate = damp_rate, mask = gaus_mask_x2, target = T_target)

gaus_mask_y1 = GaussianMask{:y}(center = Ly/2, width = Ly/32)
gaus_mask_y2 = GaussianMask{:y}(center = -Ly/2, width = Ly/32)
sponge_y_vel1 = Relaxation(; rate = damp_rate, mask = gaus_mask_y1)
sponge_y_T1 = Relaxation(; rate = damp_rate, mask = gaus_mask_y1, target = T_target)
sponge_y_vel2 = Relaxation(; rate = damp_rate, mask = gaus_mask_y2)
sponge_y_T2 = Relaxation(; rate = damp_rate, mask = gaus_mask_y2, target = T_target)
gaus = [(sponge_x_vel1, sponge_y_vel1, sponge_x_vel2, sponge_y_vel2),
        (sponge_x_T1, sponge_y_T1, sponge_x_T2, sponge_y_T2)]

# linear mask
linear_mask_x1 = PiecewiseLinearMask{:x}(center = Lx/2, width = Ly/32)
linear_mask_x2 = PiecewiseLinearMask{:x}(center = -Lx/2, width = Ly/32)
sponge_x_vel1 = Relaxation(; rate = damp_rate, mask = linear_mask_x1)
sponge_x_T1 = Relaxation(; rate = damp_rate, mask = linear_mask_x1, target = T_target)
sponge_x_S1 = Relaxation(; rate = damp_rate, mask = linear_mask_x1)
sponge_x_vel2 = Relaxation(; rate = damp_rate, mask = linear_mask_x2)
sponge_x_T2 = Relaxation(; rate = damp_rate, mask = linear_mask_x2, target = T_target)
sponge_x_S2 = Relaxation(; rate = damp_rate, mask = linear_mask_x2)

linear_mask_y1 = PiecewiseLinearMask{:y}(center = Ly/2, width = Ly/32)
sponge_y_vel1 = Relaxation(; rate = damp_rate, mask = linear_mask_y1)
sponge_y_T1 = Relaxation(; rate = damp_rate, mask = linear_mask_y1, target = T_target)
sponge_y_S1 = Relaxation(; rate = damp_rate, mask = linear_mask_y1)
linear_mask_y2 = PiecewiseLinearMask{:y}(center = -Ly/2, width = Ly/32)
sponge_y_vel2 = Relaxation(; rate = damp_rate, mask = linear_mask_y2)
sponge_y_T2 = Relaxation(; rate = damp_rate, mask = linear_mask_y2, target = T_target)
sponge_y_S2 = Relaxation(; rate = damp_rate, mask = linear_mask_y2)
linear = [(sponge_x_vel1, sponge_y_vel1, sponge_x_vel2, sponge_y_vel2),
            (sponge_x_T1, sponge_y_T1, sponge_x_T2, sponge_y_T2)]
for (i, mask) in enumerate([gaus, linear]) #[gaus, linear, nothing]
    @show mask_str[i]
    ## defining model
    if mask == nothing
        model = NonhydrostaticModel(grid;
                                    buoyancy, 
                                    advection = WENO(),
                                    tracers = (:T, :S,),
                                    timestepper = :RungeKutta3,
                                    boundary_conditions = (u = u_bcs, v = v_bcs, S=S_bcs, T=T_bcs),
                                    )
    else
        model = NonhydrostaticModel(grid;
                                    buoyancy, 
                                    advection = WENO(),
                                    tracers = (:T, :S,),
                                    timestepper = :RungeKutta3,
                                    boundary_conditions = (u = u_bcs, v = v_bcs, S=S_bcs, T=T_bcs),
                                    forcing = (T = mask[2], u = mask[1], v = mask[1], w = mask[1])
                                    )
    end
    @show model
    ## ICs
    set!(model, u=0.0, v=0.0, T=Tᵢ, S=0.0)

    # defining simulation
    simulation = Simulation(model, Δt=min_step, stop_time = 11hours) 
    @show simulation
    ## progress callback
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
    ## updating cfl every time step
    conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, diffusive_cfl = 1.0, min_Δt = min_step, max_Δt=30seconds) #ensrues cfl is updated ever iteration
    ## output files
    output_interval = 0.2hours
    u, v, w = model.velocities
    T = model.tracers.T
    S = model.tracers.S
    P_static = model.pressures.pHY′
    P_dynamic = model.pressures.pNHS
    rel_path = "localoutputs/sponge testing/$(mask_str[i])/width = $(Lx/32), rate = $(damp_rate)/"
    simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T, S, P_static, P_dynamic),
                                                        with_halos=false,
                                                        dir = rel_path, 
                                                        array_type = Array{Float64},
                                                        schedule = TimeInterval(output_interval),
                                                        filename = "fields.jld2",
                                                        overwrite_existing = true, 
                                                        init = save_grid!)

    # running the simulation
    run!(simulation)
end

