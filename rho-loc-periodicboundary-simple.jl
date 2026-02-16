using Pkg
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ
## simulation parameters
Lx = 320    # (m) domain horizontal extents
Ly = 320    # (m) domain horizontal extents
Lz = 96    # (m) domain depth 
initial_mixed_layer_depth = 30.0 # m 
dTdz = 0.01  # K m⁻¹, temperature gradient
β = 2.0e-4     # 1/K, thermal expansion coefficient

# BCs
u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
g = Oceananigans.defaults.gravitational_acceleration

u_f = 0.001
b0 = -4e-1 # m s⁻²
Jᵇ = -u_f*b0 # m² s⁻³, surface buoyancy flux
@inline function bflux_t(x, y, t) 
    if (t <= 6hours)
        σ = 10.0 # m
        return Jᵇ/(2*pi* σ^2) * exp(-(x-Lx/2)^2 / (2 * σ^2)) * exp(-(y-Ly/2)^2 / (2 * σ^2)) 
    else
        return 0.0
    end
end
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(bflux_t), 
                                    bottom = GradientBoundaryCondition(g*β*dTdz))

buoyancy = BuoyancyTracer()
## ICs
r(x, y, z) = (randn(Xoshiro())) * exp(z/4)
uᵢ(x, y, z) = u_f #* r(x, y, z)
vᵢ(x, y, z) = -u_f #* r(x, y, z)

σ = 10.0 # m
plume(x, y, z) = b0/sqrt((2*pi)^3* (σ^2)^3) * exp(-z^2 / (2 * σ^2)) * exp(-(x-Lx/2)^2 / (2 * σ^2)) * exp(-(y-Ly/2)^2 / (2 * σ^2)) 
bᵢ(x, y, z) = z > - initial_mixed_layer_depth ? 0.0 :#g*β*dTdz * Lz * 1e-6 * r(x, y, z) : #random noise in the mixed layer
                g*β*dTdz * (z + initial_mixed_layer_depth) #+ g*β*dTdz * Lz * 1e-6 * r(x, y, z)
for hor in (256, 128,)
    Nx = hor
    Ny = hor
    if hor == 256
        vert = (128, )
    elseif hor == 64
        vert = (16,)
    elseif hor == 128
        vert = (128, 256)
    end
    for Nz in vert
        println("Running simulation with Nx = $Nx, Ny = $Ny, Nz = $Nz")
        ## referring to files with desiraed functions
        grid = RectilinearGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
        @show grid
        ## defining model
        model = NonhydrostaticModel(grid;
                                    buoyancy, 
                                    advection = WENO(),
                                    tracers = (:b,),
                                    timestepper = :RungeKutta3,
                                    boundary_conditions = (u = u_bcs, v = v_bcs, b=b_bcs),
                                    )
        @show model
        set!(model, u=uᵢ, v=vᵢ, b=bᵢ)

        # defining simulation
        simulation = Simulation(model, Δt=30, stop_time = 12hours) 
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
        output_interval = 0.2hours

        path = "localoutputs/b tracer for NBP/no noise and no closure/flux b tracer Nx = $Nx, Ny = $Ny, Nz = $Nz/"
        u, v, w = model.velocities
        b = model.tracers.b
        P_static = model.pressures.pHY′
        P_dynamic = model.pressures.pNHS
        simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, b, P_static, P_dynamic),
                                                            dir = path,  with_halos=false,
                                                            array_type = Array{Float64},
                                                            schedule = TimeInterval(output_interval),
                                                            filename = "fields.jld2", #$(rank)
                                                            overwrite_existing = true)

        # running the simulation
        run!(simulation)
    end
end