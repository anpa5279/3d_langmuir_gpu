using Pkg
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BuoyancyFormulations: g_Earth
import Oceananigans.BoundaryConditions: fill_halo_regions!, PerturbationAdvectionOpenBoundaryCondition
using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ
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
## referring to files with desiraed functions
grid = RectilinearGrid(; topology =(Bounded, Bounded), size=(Nx, Nz), extent=(Lx, Lz)) #arch
## stokes drift
include("stokes.jl")
dusdz = Field{Nothing, Center}(grid)
Nx_local, Ny_local, Nz_local = size(dusdz)
z1d = grid.z.cᵃᵃᶜ[1:Nz_local]
dusdz_1d = dstokes_dz.(z1d, u₁₀)
set!(dusdz, dusdz_1d)
us = Field{Nothing, Center}(grid)
us_1d = stokes_velocity.(z1d, u₁₀)
set!(us, us_1d)
@show dusdz
stokes = UniformStokesDrift(∂z_uˢ=dusdz)

## BCs
u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(us_1d[Nz]), 
                                west   = NoSlipBoundaryCondition(),
                                east   = NoSlipBoundaryCondition())
w_bcs = FieldBoundaryConditions(bottom = NoSlipBoundaryCondition(),
                                top    = NoSlipBoundaryCondition())
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q/(ρₒ*cᴾ)),
                                bottom = GradientBoundaryCondition(dTdz))
                                


# running the simulation
run!(simulation)#; pickup = true)

function run_model2D(grid, bcs, stokes; plot=true, stop_time=3hours, name="")
    ## defining forcing (coriolis, buoyancy, etc.)
    buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = β), constant_salinity = S₀)
    ## defining model
    model = NonhydrostaticModel(; grid, buoyancy, 
                                advection = WENO(),
                                tracers = (:T),
                                timestepper = :RungeKutta3,
                                stokes_drift = stokes,
                                boundary_conditions = bcs,)
                                @show model
    ## ICs
    r(x, y, z) = randn(Xoshiro()) * exp(z/4)
    Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+dTdz * model.grid.Lz * 1e-6 * r(x, y, z)
    uᵢ(x, y, z) = u_f * r(x, y, z)
    set!(model, u=uᵢ, T=Tᵢ)
    day = 24hours
    simulation = Simulation(model, Δt=30, stop_time = 3hours) 
    ## progress function
    function progress(simulation)
        u, v, w = simulation.model.velocities
        # Print a progress message
        msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e) ms⁻¹, wall time: %s\n",
                    iteration(simulation),
                    prettytime(time(simulation)),
                    prettytime(simulation.Δt),
                    maximum(abs, u), maximum(abs, w),
                    prettytime(simulation.run_wall_time))
        @info msg
        return nothing
    end
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
    ## updating cfl every time step
    conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, max_Δt=30seconds) #ensrues cfl is updated ever iteration
    ## output files
    field_file = "localoutputs/open_fields.jld2"
    output_interval = 0.25hours
    u, w = model.velocities
    T = model.tracers.T
    P_static = model.pressures.pHY′
    P_dynamic = model.pressures.pNHS
    simulation.output_writers[:fields] = JLD2Writer(model, (; u, w, T, P_static, P_dynamic),
                                                        schedule = TimeInterval(output_interval),
                                                        filename = field_file, #$(rank)
                                                        overwrite_existing = true,
                                                        init = save_IC!)
    avg_file = "localoutputs/open_averages.jld2"
    W = Average(w, dims=(1, 2))
    U = Average(u, dims=(1, 2))
    T = Average(T, dims=(1, 2))
                                                        
    simulation.output_writers[:averages] = JLD2Writer(model, (; U, W, T),
                                                        schedule = AveragedTimeInterval(output_interval, window=output_interval),
                                                        filename = avg_file,
                                                        overwrite_existing = true)
    run!(simulation)

    if plot
        # load the results
        u_post = FieldTimeSeries(field_file, "u")
        w_post = FieldTimeSeries(field_file, "w")
        T_post = FieldTimeSeries(field_file, "T")
        static_post = FieldTimeSeries(field_file, "P_static")
        dynamic_post = FieldTimeSeries(field_file, "P_dynamic")
        @info "Loaded results"

        x, y, z = nodes(u_post, with_halos=true)

        # plot the results
        fig = Figure(size = (600, 600))
        n = Observable(1)

        w_plot = @lift w_post[:, :, $n].parent
        ax = Axis(fig[1, 1], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "w")
        heatmap!(ax, collect(x), collect(y), w_plot, colorrange = (-2, 2), colormap = :curl)

        u_plot = @lift u_post[:, :, $n].parent
        ax = Axis(fig[1, 2], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "u")
        heatmap!(ax, collect(x), collect(y), u_plot, colorrange = (-2, 2), colormap = :curl)

        T_plot = @lift T_post[:, :, $n].parent
        ax = Axis(fig[1, 2], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "T")
        heatmap!(ax, collect(x), collect(y), T_plot, colorrange = (-2, 2), colormap = :curl)

        static_plot = @lift static_post[:, :, $n].parent
        ax = Axis(fig[1, 2], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "Hydrostatic Pressure")
        heatmap!(ax, collect(x), collect(y), static_plot, colorrange = (-2, 2), colormap = :curl)

        dyn_plot = @lift dynamic_post[:, :, $n].parent
        ax = Axis(fig[1, 2], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "Hydrodynamic Pressure")
        heatmap!(ax, collect(x), collect(y), dyn_plot, colorrange = (-2, 2), colormap = :curl)

        resize_to_layout!(fig)
        record(fig, "2d_$name.mp4", 1:length(w_post.times), framerate = 16) do i;
            n[] = i
            i % 10 == 0 && @info "$(n.val) of $(length(w_post.times))"
        end
    end
end


