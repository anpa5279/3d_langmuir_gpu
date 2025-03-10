using Pkg
using CairoMakie
using Oceananigans
using JLD2

for i in 0:3
    fields_file="outputs/langmuir_turbulence_fields_$(i)_rank$(i).jld2"
    averages_file="outputs/langmuir_turbulence_averages_$(i)_rank$(i).jld2"

    fields = load(fields_file)
    averages = load(averages_file)

    #println(fields)
    #println(averages)

    time_series = (;
        w = FieldTimeSeries("outputs/langmuir_turbulence_fields_$(i)_rank$(i).jld2", "w"),
        u = FieldTimeSeries("outputs/langmuir_turbulence_fields_$(i)_rank$(i).jld2", "u"),
        B = FieldTimeSeries("outputs/langmuir_turbulence_averages_$(i)_rank$(i).jld2", "B"),
        U = FieldTimeSeries("outputs/langmuir_turbulence_averages_$(i)_rank$(i).jld2", "U"),
        V = FieldTimeSeries("outputs/langmuir_turbulence_averages_$(i)_rank$(i).jld2", "V"),
        wv = FieldTimeSeries("outputs/langmuir_turbulence_averages_$(i)_rank$(i).jld2", "wv"))
end

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
