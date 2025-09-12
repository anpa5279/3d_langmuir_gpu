using Pkg
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BuoyancyFormulations: g_Earth
import Oceananigans.BoundaryConditions: FlatExtrapolationOpenBoundaryCondition, PerturbationAdvectionOpenBoundaryCondition
using Oceananigans.Grids: topology
using CairoMakie
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
## running and plotting  function                                
function run_model2D(grid, bcs, stokes; plot=true, stop_time=3hours, name="")
    ## defining forcing (coriolis, buoyancy, etc.)
    buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = β), constant_salinity = S₀)
    ## defining model
    global model = NonhydrostaticModel(; grid, buoyancy, 
                                advection = WENO(),
                                tracers = (:T),
                                timestepper = :RungeKutta3,
                                stokes_drift = stokes,
                                boundary_conditions = bcs,)
    
    ## ICs
    r(x, z) = randn(Xoshiro(1234), (grid.Nx + grid.Nz+3))[Int(1 + round(grid.Nx*x/grid.Lx-grid.Nz*z/grid.Lz))] * exp(z / 4)
    Tᵢ(x, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+dTdz * model.grid.Lz * 1e-6 * r(x, z)
    uᵢ(x, z) = u_f * r(x, z)
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
    function save_IC!(file, model)
        file["IC/friction_velocity"] = u_f
        file["IC/stokes_velocity"] = us_top
        file["IC/wind_speed"] = u₁₀
        return nothing
    end
    field_file = "localoutputs/$(name)_fields.jld2"
    output_interval = 0.05hours
    u, w = model.velocities
    T = model.tracers.T
    P_static = model.pressures.pHY′
    P_dynamic = model.pressures.pNHS
    simulation.output_writers[:fields] = JLD2Writer(model, (; u, w, T, P_static, P_dynamic),
                                                        schedule = TimeInterval(output_interval),
                                                        filename = field_file, #$(rank)
                                                        overwrite_existing = true,
                                                        init = save_IC!)
    avg_file = "localoutputs/$(name)_averages.jld2"
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
        fig = Figure(size = (850, 850))
        n = Observable(1)

        w_plot = @lift w_post[:, 1, :, $n].parent
        ax = Axis(fig[1, 1], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "w")
        wpl = heatmap!(ax, collect(x), collect(z), w_plot, colorrange = (-0.0005, 0.0005), colormap = :balance)
        Colorbar(fig[1, 2], wpl; label = "m s⁻¹")

        u_plot = @lift u_post[:, 1, :, $n].parent
        ax = Axis(fig[1, 5], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "u")
        upl = heatmap!(ax, collect(x), collect(z), u_plot, colorrange = (-0.1, 0.1), colormap = :balance)
        Colorbar(fig[1, 6], upl; label = "m s⁻¹")

        T_plot = @lift T_post[:, 1, :, $n].parent
        ax = Axis(fig[2, 1], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "T")
        Tpl = heatmap!(ax, collect(x), collect(z), T_plot, colorrange = (24, 25), colormap = :curl)
        Colorbar(fig[2, 2], Tpl; label = "C")

        static_plot = @lift static_post[:, 1, :, $n].parent
        ax = Axis(fig[2, 3], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "Hydrostatic Pressure")
        hPpl = heatmap!(ax, collect(x), collect(z), static_plot, colorrange = (-4, 4), colormap = :balance)
        Colorbar(fig[2, 4], hPpl, label = "Pa")

        dyn_plot = @lift dynamic_post[:, 1, :, $n].parent
        ax = Axis(fig[2, 5], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "Hydrodynamic Pressure")
        nPpl = heatmap!(ax, collect(x), collect(z), dyn_plot, colorrange = (-2, 2), colormap = :balance)
        Colorbar(fig[2, 6], nPpl, label = "Pa")

        resize_to_layout!(fig)
        record(fig, "localoutputs/$name.mp4", 1:length(w_post.times), framerate = 16) do i;
            n[] = i
        end
    end
end
function run_model3D(grid, bcs, stokes; plot=true, stop_time=3hours, name="")
    ## defining forcing (coriolis, buoyancy, etc.)
    buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = β), constant_salinity = S₀)
    ## defining model
    global model = NonhydrostaticModel(; grid, buoyancy, 
                                advection = WENO(),
                                tracers = (:T),
                                timestepper = :RungeKutta3,
                                stokes_drift = stokes,
                                boundary_conditions = bcs,)
                                
    ## ICs
    r(x, y, z) = randn(Xoshiro(1234), (grid.Nx + grid.Ny + grid.Nz+3))[Int(1 + round(grid.Nx*x/grid.Lx+grid.Ny*y/grid.Ly-grid.Nz*z/grid.Lz))] * exp(z / 4)
    Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+dTdz * model.grid.Lz * 1e-6 * r(x, y, z)
    uᵢ(x, y, z) = u_f * r(x, y, z)
    vᵢ(x, y, z) = -u_f * r(x, y, z)
    set!(model, u=uᵢ, v=vᵢ, T=Tᵢ)
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
    function save_IC!(file, model)
        file["IC/friction_velocity"] = u_f
        file["IC/stokes_velocity"] = us_top
        file["IC/wind_speed"] = u₁₀
        return nothing
    end
    field_file = "localoutputs/$(name)_fields.jld2"
    output_interval = 0.25hours
    u, v, w = model.velocities
    T = model.tracers.T
    P_static = model.pressures.pHY′
    P_dynamic = model.pressures.pNHS
    simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T, P_static, P_dynamic),
                                                        schedule = TimeInterval(output_interval),
                                                        filename = field_file,
                                                        overwrite_existing = true,
                                                        init = save_IC!)
    avg_file = "localoutputs/$(name)_averages.jld2"
    W = Average(w, dims=(1, 2))
    U = Average(u, dims=(1, 2))
    V = Average(v, dims=(1, 2))
    T = Average(T, dims=(1, 2))
                                                        
    simulation.output_writers[:averages] = JLD2Writer(model, (; U, V, W, T),
                                                        schedule = AveragedTimeInterval(output_interval, window=output_interval),
                                                        filename = avg_file,
                                                        overwrite_existing = true)
    run!(simulation)

    if plot
        # load the results
        u_post = FieldTimeSeries(field_file, "u")
        v_post = FieldTimeSeries(field_file, "v")
        w_post = FieldTimeSeries(field_file, "w")
        T_post = FieldTimeSeries(field_file, "T")
        static_post = FieldTimeSeries(field_file, "P_static")
        dynamic_post = FieldTimeSeries(field_file, "P_dynamic")
        @info "Loaded results"

        x, y, z = nodes(u_post, with_halos=true)

        # plot the results
        fig = Figure(size = (850, 850))
        n = Observable(1)

        w_plot = @lift w_post[:, 1, :, $n].parent
        ax = Axis(fig[1, 1], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "w")
        wpl = heatmap!(ax, collect(x), collect(z), w_plot, colorrange = (-0.0005, 0.0005), colormap = :balance)
        Colorbar(fig[1, 2], wpl; label = "m s⁻¹")

        v_plot = @lift v_post[:, 1, :, $n].parent
        ax = Axis(fig[1, 3], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "v")
        vpl = heatmap!(ax, collect(x), collect(z), v_plot, colorrange = (-0.01, 0.01), colormap = :balance)
        Colorbar(fig[1, 4], vpl; label = "m s⁻¹")

        
        u_plot = @lift u_post[:, 1, :, $n].parent
        ax = Axis(fig[1, 5], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "u")
        upl = heatmap!(ax, collect(x), collect(z), u_plot, colorrange = (-0.1, 0.1), colormap = :balance)
        Colorbar(fig[1, 6], upl; label = "m s⁻¹")

        T_plot = @lift T_post[:, 1, :, $n].parent
        ax = Axis(fig[2, 1], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "T")
        Tpl = heatmap!(ax, collect(x), collect(z), T_plot, colorrange = (24, 25), colormap = :curl)
        Colorbar(fig[2, 2], Tpl; label = "C")

        static_plot = @lift static_post[:, 1, :, $n].parent
        ax = Axis(fig[2, 3], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "Hydrostatic Pressure")
        hPpl = heatmap!(ax, collect(x), collect(z), static_plot, colorrange = (-4, 4), colormap = :balance)
        Colorbar(fig[2, 4], hPpl, label = "Pa")

        dyn_plot = @lift dynamic_post[:, 1, :, $n].parent
        ax = Axis(fig[2, 5], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = Lx, height = Lz, title = "Hydrodynamic Pressure")
        nPpl = heatmap!(ax, collect(x), collect(z), dyn_plot, colorrange = (-2, 2), colormap = :balance)
        Colorbar(fig[2, 6], nPpl, label = "Pa")

        resize_to_layout!(fig)
        record(fig, "localoutputs/$name.mp4", 1:length(w_post.times), framerate = 16) do i;
            n[] = i
        end
    end
end
## grids 
grid2d = RectilinearGrid(; topology =(Bounded, Flat, Bounded), size=(Nx, Nz), extent=(Lx, Lz)) #arch
grid3d = RectilinearGrid(; topology =(Bounded, Bounded, Bounded), size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)) #arch
grid3d_periodic = RectilinearGrid(; topology =(Periodic, Periodic, Bounded), size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)) #arch

## stokes drift
include("stokes.jl")
dusdz_2d = Field{Nothing, Nothing, Center}(grid2d)
dusdz_3d = Field{Nothing, Nothing, Center}(grid3d)
Nx_local, Ny_local, Nz_local = size(dusdz_2d)
z1d = grid2d.z.cᵃᵃᶜ[1:Nz_local]
dusdz_1d = dstokes_dz.(z1d, u₁₀)
set!(dusdz_2d, dusdz_1d)
set!(dusdz_3d, dusdz_1d)
us_1d = stokes_velocity.(z1d, u₁₀)
stokes2d = UniformStokesDrift(∂z_uˢ=dusdz_2d)
stokes3d = UniformStokesDrift(∂z_uˢ=dusdz_3d)

## setting up varied BCs
us_top = us_1d[Nz]
u_f = La_t^2 * us_top
τx = -(u_f^2)
@inline u∞(y, t, p) = p.U * cos(t * 2π / p.TT) * (1 + 0.01 * randn())
@inline u∞(y, z, t, p) = p.U * cos(t * 2π / p.TT) * (1 + 0.01 * randn())
@inline v∞(x, t, p) = p.U * sin(t * 2π / p.TT) * (1 + 0.01 * randn())
@inline v∞(x, z, t, p) = p.U * sin(t * 2π / p.TT) * (1 + 0.01 * randn())
inflow_timescale = outflow_timescale = 1/4
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q/(ρₒ*cᴾ)),
                                bottom = GradientBoundaryCondition(0.0))
u_val = ValueBoundaryCondition(us_1d[Nz])
u_flux = FluxBoundaryCondition(τx)
no_slip_bc = ValueBoundaryCondition(0.0)
TT = 50
u_fe = FlatExtrapolationOpenBoundaryCondition(u∞, parameters = (; U = us_top, TT = TT), relaxation_timescale = 1)
v_fe = FlatExtrapolationOpenBoundaryCondition(v∞, parameters = (; U = us_top, TT = TT), relaxation_timescale = 1)
w_fe = FlatExtrapolationOpenBoundaryCondition(v∞, parameters = (; U = us_top, TT = TT), relaxation_timescale = 1)
w_fe = FlatExtrapolationOpenBoundaryCondition(v∞, parameters = (; U = us_top, TT = TT), relaxation_timescale = 1)

u_boundaries_fe_val = FieldBoundaryConditions(top = u_val, west = u_fe, east = u_fe)
u_boundaries_fe_flux = FieldBoundaryConditions(top = u_flux, west = u_fe, east = u_fe)
u_boundaries_fe_val_couette = FieldBoundaryConditions(top = u_val, bottom = no_slip_bc, west = u_fe, east = u_fe)
u_boundaries_fe_flux_couette = FieldBoundaryConditions(top = u_flux, bottom = no_slip_bc, west = u_fe, east = u_fe)
v_boundaries_fe = FieldBoundaryConditions(south = v_fe, north = v_fe)
w_boundaries_fe = FieldBoundaryConditions(bottom = w_fe, top = w_fe)
feobcs_val = (u = u_boundaries_fe_val, v = v_boundaries_fe, w = w_boundaries_fe, T = T_bcs)
feobcs_flux = (u = u_boundaries_fe_flux, v = v_boundaries_fe, w = w_boundaries_fe, T = T_bcs)
feobcs_couette_val = (u = u_boundaries_fe_val_couette, v = v_boundaries_fe, w = w_boundaries_fe, T = T_bcs)
feobcs_couette_flux = (u = u_boundaries_fe_flux_couette, v = v_boundaries_fe, w = w_boundaries_fe, T = T_bcs)

u_boundaries_pa_val = FieldBoundaryConditions(top = u_val, 
                                            west   = PerturbationAdvectionOpenBoundaryCondition(u∞; parameters = (; U = us_top, TT = TT), inflow_timescale, outflow_timescale),
                                            east   = PerturbationAdvectionOpenBoundaryCondition(u∞; parameters = (; U = us_top, TT = TT), inflow_timescale, outflow_timescale))
u_boundaries_pa_flux = FieldBoundaryConditions(top = u_flux, 
                                            west   = PerturbationAdvectionOpenBoundaryCondition(u∞; parameters = (; U = us_top, TT = TT), inflow_timescale, outflow_timescale),
                                            east   = PerturbationAdvectionOpenBoundaryCondition(u∞; parameters = (; U = us_top, TT = TT), inflow_timescale, outflow_timescale))
u_boundaries_pa_val_couette = FieldBoundaryConditions(top = u_val, bottom = no_slip_bc,
                                            west   = PerturbationAdvectionOpenBoundaryCondition(u∞; parameters = (; U = us_top, TT = TT), inflow_timescale, outflow_timescale),
                                            east   = PerturbationAdvectionOpenBoundaryCondition(u∞; parameters = (; U = us_top, TT = TT), inflow_timescale, outflow_timescale))
u_boundaries_pa_flux_couette = FieldBoundaryConditions(top = u_flux, bottom = no_slip_bc,
                                            west   = PerturbationAdvectionOpenBoundaryCondition(u∞; parameters = (; U = us_top, TT = TT), inflow_timescale, outflow_timescale),
                                            east   = PerturbationAdvectionOpenBoundaryCondition(u∞; parameters = (; U = us_top, TT = TT), inflow_timescale, outflow_timescale))
v_boundaries_pa = FieldBoundaryConditions(south  = PerturbationAdvectionOpenBoundaryCondition(v∞; parameters = (; U = us_top, TT = TT), inflow_timescale, outflow_timescale),
                                            north  = PerturbationAdvectionOpenBoundaryCondition(v∞; parameters = (; U = us_top, TT = TT), inflow_timescale, outflow_timescale))
w_boundaries_pa = FieldBoundaryConditions(bottom = PerturbationAdvectionOpenBoundaryCondition(v∞; parameters = (; U = us_top, TT = TT), inflow_timescale, outflow_timescale),
                                            top    = PerturbationAdvectionOpenBoundaryCondition(v∞; parameters = (; U = us_top, TT = TT), inflow_timescale, outflow_timescale))
paobcs_val = (u = u_boundaries_pa_val, v = v_boundaries_pa, w = w_boundaries_pa, T = T_bcs)
paobcs_flux = (u = u_boundaries_pa_flux, v = v_boundaries_pa, w = w_boundaries_pa, T = T_bcs)
paobcs_couette_val = (u = u_boundaries_pa_val_couette, v = v_boundaries_pa, w = w_boundaries_pa, T = T_bcs)
paobcs_couette_flux = (u = u_boundaries_pa_flux_couette, v = v_boundaries_pa, w = w_boundaries_pa, T = T_bcs)

u_fe_lid = FlatExtrapolationOpenBoundaryCondition(0.0; relaxation_timescale = 1)
v_fe_lid = FlatExtrapolationOpenBoundaryCondition(0.0; relaxation_timescale = 1)
w_fe_lid = FlatExtrapolationOpenBoundaryCondition(0.0; relaxation_timescale = 1)

u_boundaries_fe_val_lid = FieldBoundaryConditions(top = u_val, west = u_fe_lid, east = u_fe_lid, north = u_fe_lid, south = u_fe_lid)
u_boundaries_fe_flux_lid = FieldBoundaryConditions(top = u_flux, west = u_fe_lid, east = u_fe_lid, north = u_fe_lid, south = u_fe_lid)
v_boundaries_fe_lid = FieldBoundaryConditions(v_fe_lid)
w_boundaries_fe_lid = FieldBoundaryConditions(w_fe_lid)
feobcs_val_lid = (u = u_boundaries_fe_val_lid, v = v_boundaries_fe_lid, w = w_boundaries_fe_lid, T = T_bcs)
feobcs_flux_lid = (u = u_boundaries_fe_flux_lid, v = v_boundaries_fe_lid, w = w_boundaries_fe_lid, T = T_bcs)

u_boundaries_pa_val_lid = FieldBoundaryConditions(top = u_val, 
                                            west   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale),
                                            east   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale),
                                            north   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale),
                                            south   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale))
u_boundaries_pa_flux_lid = FieldBoundaryConditions(top = u_flux, 
                                            west   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale),
                                            east   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale),
                                            north   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale),
                                            south   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale))
u_boundaries_pa_val_couette_lid = FieldBoundaryConditions(top = u_val, bottom = no_slip_bc,
                                            west   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale),
                                            east   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale),
                                            north   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale),
                                            south   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale))
u_boundaries_pa_flux_couette_lid = FieldBoundaryConditions(top = u_flux, bottom = no_slip_bc,
                                            west   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale),
                                            east   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale),
                                            north   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale),
                                            south   = PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale))
v_boundaries_pa_lid = FieldBoundaryConditions(PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale))
w_boundaries_pa_lid = FieldBoundaryConditions(PerturbationAdvectionOpenBoundaryCondition(0.0; inflow_timescale, outflow_timescale))
paobcs_val_lid = (u = u_boundaries_pa_val_lid, v = v_boundaries_pa_lid, w = w_boundaries_pa_lid, T = T_bcs)
paobcs_flux_lid = (u = u_boundaries_pa_flux_lid, v = v_boundaries_pa_lid, w = w_boundaries_pa_lid, T = T_bcs)

u_val_lid = FieldBoundaryConditions(top = u_val, bottom = no_slip_bc, north = no_slip_bc, south = no_slip_bc)
u_flux_lid = FieldBoundaryConditions(top = u_flux, bottom = no_slip_bc, north = no_slip_bc, south = no_slip_bc)
v_lid = FieldBoundaryConditions(no_slip_bc)
w_lid = FieldBoundaryConditions(no_slip_bc)
val_lid = (u = u_val_lid, v = v_lid, w = w_lid, T = T_bcs)
flux_lid = (u = u_flux_lid, v = v_lid, w = w_lid, T = T_bcs)

bc_options = (feobcs_val, feobcs_couette_val, paobcs_val, paobcs_couette_val, 
            feobcs_flux, feobcs_couette_flux, paobcs_flux, paobcs_couette_flux, 
            feobcs_val_lid, paobcs_val_lid, val_lid,
            feobcs_flux_lid, paobcs_flux_lid, flux_lid)
loc_lid = length(bc_options) - 6
## naming function
matching_scheme_name(obc) = string(nameof(typeof(obc.classification)))
matching_scheme_sides(obc) = string(nameof(typeof(obc.classification.matching_scheme)))

## running all combinations of cases
for grid in (grid2d, grid3d, grid3d_periodic) 
    if topology(grid)[2] == Flat
        dim = "2d"
        bc_oi = bc_options
    elseif topology(grid) == (Bounded, Bounded, Bounded)
        dim = "3d_open"
        bc_oi = bc_options
    else
        dim = "3d_periodic"
        bc_oi = bc_options[1:(length(bc_options)-loc_lid)] #no lid cases for periodic
    end
    d = 1
    for bcs in bc_oi
        if topology(grid)[1] == Periodic #3d periodic case
            boundary_conditions = (u = FieldBoundaryConditions(top = bcs.u.top, bottom = bcs.u.bottom), 
                                v = FieldBoundaryConditions(top = bcs.v.top, bottom = bcs.v.bottom),  
                                w = FieldBoundaryConditions(top = bcs.w.top, bottom = bcs.w.bottom), 
                                T = bcs.T)
            boundary_conditions_order = (w = boundary_conditions[:w], u = boundary_conditions[:u], v = boundary_conditions[:v], T = boundary_conditions[:T])
            run_name = dim * "_utop_" * matching_scheme_name(boundary_conditions.u.top)
        elseif topology(grid) == (Bounded, Bounded, Bounded) && typeof(bcs.w.east) !== Oceananigans.BoundaryConditions.DefaultBoundaryCondition{BoundaryCondition{Oceananigans.BoundaryConditions.Value, Float64}} #3d open case
            boundary_conditions = bcs
            boundary_conditions_order = (w = boundary_conditions[:w], u = boundary_conditions[:u], v = boundary_conditions[:v], T = boundary_conditions[:T])
            run_name = dim * "_sides_" * matching_scheme_sides(boundary_conditions.u.east) * "_utop_" * matching_scheme_name(boundary_conditions.u.top)
        elseif topology(grid) == (Bounded, Bounded, Bounded) && typeof(bcs.w.east) == Oceananigans.BoundaryConditions.DefaultBoundaryCondition{BoundaryCondition{Oceananigans.BoundaryConditions.Value, Float64}} #3d open case, lid val bcs
            boundary_conditions = bcs
            boundary_conditions_order = (w = boundary_conditions[:w], u = boundary_conditions[:u], v = boundary_conditions[:v], T = boundary_conditions[:T])
            run_name = dim * "_sides_Value_utop_" * matching_scheme_name(boundary_conditions.u.top)
        elseif topology(grid)[2] == Flat && typeof(bcs.w.east) !== Oceananigans.BoundaryConditions.DefaultBoundaryCondition{BoundaryCondition{Oceananigans.BoundaryConditions.Value, Float64}} #2d bcs
            boundary_conditions = (u = FieldBoundaryConditions(top = bcs.u.top, bottom = bcs.u.bottom, east = bcs.u.east, west = bcs.u.west), w = bcs.w, T = bcs.T)
            boundary_conditions_order = (w = boundary_conditions[:w], u = boundary_conditions[:u], T = boundary_conditions[:T])
            run_name = dim * "_sides_" * matching_scheme_sides(boundary_conditions.u.east) * "_utop_" * matching_scheme_name(boundary_conditions.u.top)
        elseif topology(grid)[2] == Flat && typeof(bcs.w.east) == Oceananigans.BoundaryConditions.DefaultBoundaryCondition{BoundaryCondition{Oceananigans.BoundaryConditions.Value, Float64}} #2d lid val bcs
            boundary_conditions = (u = FieldBoundaryConditions(top = bcs.u.top, bottom = bcs.u.bottom, east = bcs.u.east, west = bcs.u.west), w = bcs.w, T = bcs.T)
            boundary_conditions_order = (w = boundary_conditions[:w], u = boundary_conditions[:u], T = boundary_conditions[:T])
            run_name = dim * "_sides_Value_utop_" * matching_scheme_name(boundary_conditions.u.top)
        else
            boundary_conditions = (u = FieldBoundaryConditions(top = bcs.u.top, bottom = bcs.u.bottom, east = bcs.u.east, west = bcs.u.west), w = bcs.w, T = bcs.T)
            boundary_conditions_order = (w = boundary_conditions[:w], u = boundary_conditions[:u], v = boundary_conditions[:v], T = boundary_conditions[:T])
            run_name = dim * "_sides_Value_utop_" * matching_scheme_name(boundary_conditions.u.top)
        end
        @show d, run_name
        if d > loc_lid && topology(grid)[1] !== Periodic
            run_name = "lid_" * run_name
        elseif typeof(boundary_conditions.u.bottom) == BoundaryCondition{Oceananigans.BoundaryConditions.Value, Float64} && (d < loc_lid + 1) && topology(grid)[1] !== Periodic
            run_name = "couette_" * run_name
        end
        run_name1 = run_name
        for bc_order in (boundary_conditions, boundary_conditions_order)
            for stokes in (nothing, stokes3d)
                if stokes !== nothing 
                    run_name = run_name * "_stokes"
                end
                @info "Running $run_name"
                if topology(grid)[2] == Flat
                    run_model2D(grid, bc_order, stokes; name=run_name)
                else
                    run_model3D(grid, bc_order, stokes; name=run_name) 
                end
            end
            run_name = "bc_order_changed_" * run_name1
        end
        run_name = ""
        d += 1
    end
end