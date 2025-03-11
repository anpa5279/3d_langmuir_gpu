using Pkg
using CairoMakie
using Oceananigans
using Oceananigans.Units: minute, minutes, hours
using JLD2

# running locally: using Pkg; Pkg.add("Oceananigans"); Pkg.add("CairoMakie"); Pkg.add("JLD2")
times = Float64[]

for i in 0:3
    println("Loading rank $i")

    fld_file="outputs/langmuir_turbulence_fields_$(i)_rank$(i).jld2"
    averages_file="outputs/langmuir_turbulence_averages_$(i)_rank$(i).jld2"

    f = jldopen(fld_file)
    a = jldopen(averages_file)

    loc = collect(keys(f["timeseries"]["t"]))

    if i == 0
        times = f["timeseries"]["t"][loc[end]]
        p = i
        println("First time: ", times)
    end 

    for j in 1:length(loc)
        println(j)
        #println(f["timeseries"]["w"][loc[j]])
        push!(w[j + p], f["timeseries"]["w"][loc[j]])
        push!(u[j + p], f["timeseries"]["u"][loc[j]])
        push!(B[j + p], a["timeseries"]["B"][loc[j]])
        push!(U[j + p], a["timeseries"]["U"][loc[j]])
        push!(V[j + p], a["timeseries"]["V"][loc[j]])
        push!(wu[j + p], a["timeseries"]["wu"][loc[j]])
        push!(wv[j + p], a["timeseries"]["wv"][loc[j]])
    end

    p = length(loc) + p * i

    close(f)
    close(a)
    
end

times = w.times

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
            limits = ((0, Lx), (0, Ly)),
            title = wxy_title)

ax_wxz = Axis(fig[2, 1:2];
            xlabel = "x (m)",
            ylabel = "z (m)",
            aspect = AxisAspect(2),
            limits = ((0, Lx), (-Lz, 0)),
            title = wxz_title)

ax_uxz = Axis(fig[3, 1:2];
            xlabel = "x (m)",
            ylabel = "z (m)",
            aspect = AxisAspect(2),
            limits = ((0, Lx), (-Lz, 0)),
            title = uxz_title)


wₙ = @lift gridw[$n]
uₙ = @lift gridu[$n]
Bₙ = @lift view(gridB[$n], 1, 1, :)
Uₙ = @lift view(gridU[$n], 1, 1, :)
Vₙ = @lift view(gridV[$n], 1, 1, :)
wuₙ = @lift view(gridwu[$n], 1, 1, :)
wvₙ = @lift view(gridwv[$n], 1, 1, :)

k = searchsortedfirst(znodes(grid, Face(); with_halos=true), -8)
wxyₙ = @lift view(gridw[$n], :, :, k)
wxzₙ = @lift view(gridw[$n], :, 1, :)
uxzₙ = @lift view(gridu[$n], :, 1, :)

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
