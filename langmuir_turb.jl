using Pkg
using MPI
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.Units: minute, minutes, hours
using Oceananigans.BuoyancyFormulations: g_Earth
using Printf
using CairoMakie

mutable struct Params
    Nx::Int
    Ny::Int     # number of points in each of horizontal directions
    Nz::Int     # number of points in the vertical direction
    Lx::Float64
    Ly::Float64     # (m) domain horizontal extents
    Lz::Float64     # (m) domain depth 
    amplitude::Float64 # m
    wavelength::Float64 # m
    τx::Float64 # m² s⁻², surface kinematic momentum flux
    Jᵇ::Float64 # m² s⁻³, surface buoyancy flux
    N²::Float64 # s⁻², initial and bottom buoyancy gradient
    initial_mixed_layer_depth::Float64 #m
end

#defaults, these can be changed directly below
params = Params(32, 32, 32, 128.0, 128.0,64.0, 0.8, 60.0, -3.72e-5, 2.307e-8, 1.936e-5, 33.0)

# Automatically distributes among available processors
arch = Distributed(GPU())
rank = arch.local_rank
Nranks = MPI.Comm_size(arch.communicator)
println("Hello from process $rank out of $Nranks")

grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), extent=(params.Lx, params.Ly, params.Lz))
@show grid

const wavenumber = 2π / params.wavelength # m⁻¹
const frequency = sqrt(g_Earth * wavenumber) # s⁻¹
@show wavenumber, frequency

# The vertical scale over which the Stokes drift of a monochromatic surface wave
# decays away from the surface is `1/2wavenumber`, or
const vertical_scale = params.wavelength / 4π
@show vertical_scale

# Stokes drift velocity at the surface
const Uˢ = params.amplitude^2 * wavenumber * frequency # m s⁻¹
@show Uˢ

@inline uˢ(z) = Uˢ * exp(z / vertical_scale)

@inline ∂z_uˢ(z, t) = 1 / vertical_scale * Uˢ * exp(z / vertical_scale)

u_boundary_conditions = FieldBoundaryConditions(top = FluxBoundaryCondition(params.τx))
@show u_boundary_conditions

b_boundary_conditions = FieldBoundaryConditions(top = FluxBoundaryCondition(params.Jᵇ),
                                                bottom = GradientBoundaryCondition(params.N²))
@show b_boundary_conditions

coriolis = FPlane(f=1e-4) # s⁻¹

model = NonhydrostaticModel(; grid, coriolis,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            boundary_conditions = (u=u_boundary_conditions, b=b_boundary_conditions))
@show model

@inline Ξ(z) = randn() * exp(z / 4)

@inline stratification(z) = z < - params.initial_mixed_layer_depth ? params.N² * z : params.N² * (-params.initial_mixed_layer_depth)

@inline bᵢ(x, y, z) = stratification(z) + 1e-1 * Ξ(z) * params.N² * model.grid.Lz

u★ = sqrt(abs(params.τx))
@inline uᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
@inline wᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, b=bᵢ)

simulation = Simulation(model, Δt=45.0, stop_time=4hours)
@show simulation

conjure_time_step_wizard!(simulation, cfl=1.0, max_Δt=1minute)

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

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

output_interval = 5minutes

fields_to_output = merge(model.velocities, model.tracers, (; νₑ=model.diffusivity_fields.νₑ))
simulation.output_writers[:fields] =
    JLD2OutputWriter(model, fields_to_output,
                        schedule = TimeInterval(output_interval),
                        filename = "langmuir_turbulence_fields_$rank.jld2",
                        overwrite_existing = true)

u, v, w = model.velocities
b = model.tracers.b


    U = Average(u, dims=(1, 2))
    V = Average(v, dims=(1, 2))
    B = Average(b, dims=(1, 2))
wu = Average(w * u, dims=(1, 2))
wv = Average(w * v, dims=(1, 2))

if rank == 0
simulation.output_writers[:averages] =
    JLD2OutputWriter(model, (; U, V, B, wu, wv),
                        schedule = AveragedTimeInterval(output_interval, window=2minutes),
                        filename = "langmuir_turbulence_averages_$rank.jld2",
                        overwrite_existing = true)
end
run!(simulation)

time_series = (;
    w = FieldTimeSeries("langmuir_turbulence_fields.jld2", "w"),
    u = FieldTimeSeries("langmuir_turbulence_fields.jld2", "u"),
    B = FieldTimeSeries("langmuir_turbulence_averages.jld2", "B"),
    U = FieldTimeSeries("langmuir_turbulence_averages.jld2", "U"),
    V = FieldTimeSeries("langmuir_turbulence_averages.jld2", "V"),
    wu = FieldTimeSeries("langmuir_turbulence_averages.jld2", "wu"),
    wv = FieldTimeSeries("langmuir_turbulence_averages.jld2", "wv"))
                        
times = time_series.w.times

n = Observable(1)

wxy_title = @lift string("w(x, y, t) at z=-8 m and t = ", prettytime(times[$n]))
wxz_title = @lift string("w(x, z, t) at y=0 m and t = ", prettytime(times[$n]))
uxz_title = @lift string("u(x, z, t) at y=0 m and t = ", prettytime(times[$n]))

fig = Figure(size = (850, 850))

ax_B = Axis(fig[1, 4];
            xlabel = "Buoyancy (m s⁻²)",
            ylabel = "z (m)")

ax_U = Axis(fig[2, 4];
            xlabel = "Velocities (m s⁻¹)",
            ylabel = "z (m)",
            limits = ((-0.07, 0.07), nothing))

ax_fluxes = Axis(fig[3, 4];
                xlabel = "Momentum fluxes (m² s⁻²)",
                ylabel = "z (m)",
                limits = ((-3.5e-5, 3.5e-5), nothing))

ax_wxy = Axis(fig[1, 1:2];
            xlabel = "x (m)",
            ylabel = "y (m)",
            aspect = DataAspect(),
            limits = ((0, grid.Lx), (0, grid.Ly)),
            title = wxy_title)

ax_wxz = Axis(fig[2, 1:2];
            xlabel = "x (m)",
            ylabel = "z (m)",
            aspect = AxisAspect(2),
            limits = ((0, grid.Lx), (-grid.Lz, 0)),
            title = wxz_title)

ax_uxz = Axis(fig[3, 1:2];
            xlabel = "x (m)",
            ylabel = "z (m)",
            aspect = AxisAspect(2),
            limits = ((0, grid.Lx), (-grid.Lz, 0)),
            title = uxz_title)


wₙ = @lift time_series.w[$n]
uₙ = @lift time_series.u[$n]
Bₙ = @lift view(time_series.B[$n], 1, 1, :)
Uₙ = @lift view(time_series.U[$n], 1, 1, :)
Vₙ = @lift view(time_series.V[$n], 1, 1, :)
wuₙ = @lift view(time_series.wu[$n], 1, 1, :)
wvₙ = @lift view(time_series.wv[$n], 1, 1, :)

k = searchsortedfirst(znodes(grid, Face(); with_halos=true), -8)
wxyₙ = @lift view(time_series.w[$n], :, :, k)
wxzₙ = @lift view(time_series.w[$n], :, 1, :)
uxzₙ = @lift view(time_series.u[$n], :, 1, :)

wlims = (-0.03, 0.03)
ulims = (-0.05, 0.05)

lines!(ax_B, Bₙ)

lines!(ax_U, Uₙ; label = L"\bar{u}")
lines!(ax_U, Vₙ; label = L"\bar{v}")
axislegend(ax_U; position = :rb)

lines!(ax_fluxes, wuₙ; label = L"mean $wu$")
lines!(ax_fluxes, wvₙ; label = L"mean $wv$")
axislegend(ax_fluxes; position = :rb)

hm_wxy = heatmap!(ax_wxy, wxyₙ;
                colorrange = wlims,
                colormap = :balance)

Colorbar(fig[1, 3], hm_wxy; label = "m s⁻¹")

hm_wxz = heatmap!(ax_wxz, wxzₙ;
                colorrange = wlims,
                colormap = :balance)

Colorbar(fig[2, 3], hm_wxz; label = "m s⁻¹")

ax_uxz = heatmap!(ax_uxz, uxzₙ;
                colorrange = ulims,
                colormap = :balance)

Colorbar(fig[3, 3], ax_uxz; label = "m s⁻¹")

fig

frames = 1:length(times)

record(fig, "langmuir_turbulence.mp4", frames, framerate=8) do i
    n[] = i
end
