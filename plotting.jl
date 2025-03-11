using Pkg
using CairoMakie
using Oceananigans
using Oceananigans.Units: minute, minutes, hours
using JLD2

# running locally: using Pkg; Pkg.add("Oceananigans"); Pkg.add("CairoMakie"); Pkg.add("JLD2")
Nranks = 1

for i in 0:Nranks-1
    println("Loading rank $i")

    fld_file="outputs/langmuir_turbulence_fields_$(i)_rank$(i).jld2"
    averages_file="outputs/langmuir_turbulence_averages_$(i)_rank$(i).jld2"

    f = jldopen(fld_file)
    a = jldopen(averages_file)

    loc = collect(keys(f["timeseries"]["t"]))

    if i == 0
        global grid = RectilinearGrid(size = (Nranks * f["grid"]["Nx"], f["grid"]["Ny"], f["grid"]["Nz"]), extent = (Nranks * f["grid"]["Lx"], f["grid"]["Ly"], f["grid"]["Lz"]))
        global Lx = Nranks * f["grid"]["Lx"]
        global Ly = f["grid"]["Ly"]
        global Lz = f["grid"]["Lz"]
        global times = Vector{Float64}(undef, length(loc))
        global w = Array{Float64}(undef, Nranks * f["grid"]["Nx"], f["grid"]["Ny"], f["grid"]["Nz"], length(loc))
        global u = []
        global B = []
        global U = []
        global V = []
        global wu = []
        global wv = []
        for j in 1:length(loc)
            println(f["timeseries"]["t"][loc[j]])
            times[j] = f["timeseries"]["t"][loc[j]]
            w[:, :, :, j] = f["timeseries"]["w"][loc[j]]
            #push!(times, f["timeseries"]["t"][loc[j]])
            #push!(w, f["timeseries"]["w"][loc[j]])
            push!(u, f["timeseries"]["u"][loc[j]])
            push!(B, a["timeseries"]["B"][loc[j]])
            push!(U, a["timeseries"]["U"][loc[j]])
            push!(V, a["timeseries"]["V"][loc[j]])
            push!(wu, a["timeseries"]["wu"][loc[j]])
            push!(wv, a["timeseries"]["wv"][loc[j]])
        end
    else 
        for j in 1:length(loc)
            push!(w, f["timeseries"]["w"][loc[j]])
            push!(u, f["timeseries"]["u"][loc[j]])
            push!(B, a["timeseries"]["B"][loc[j]])
            push!(U, a["timeseries"]["U"][loc[j]])
            push!(V, a["timeseries"]["V"][loc[j]])
            push!(wu, a["timeseries"]["wu"][loc[j]])
            push!(wv, a["timeseries"]["wv"][loc[j]])
        end
    end 

    close(f)
    close(a)
    
end
println(size(w))
println(grid)
println(Lx)
println(size(times))

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


wₙ = @lift w[$n]
uₙ = @lift u[$n]
Bₙ = @lift view(B[$n], 1, 1, :)
Uₙ = @lift view(U[$n], 1, 1, :)
Vₙ = @lift view(V[$n], 1, 1, :)
wuₙ = @lift view(wu[$n], 1, 1, :)
wvₙ = @lift view(wv[$n], 1, 1, :)

k = searchsortedfirst(znodes(grid, Face(); with_halos=true), -8)
wxyₙ = @lift view(w[$n], :, :, k)
wxzₙ = @lift view(w[$n], :, 1, :)
uxzₙ = @lift view(u[$n], :, 1, :)

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
