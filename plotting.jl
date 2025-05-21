using Pkg
using OffsetArrays
using Statistics
using CairoMakie
using Printf
using JLD2
using Oceananigans
using Oceananigans.Units: minute, minutes, hours
using Oceananigans.BuoyancyFormulations: g_Earth

function VKE(a, u_f)
    nt = length(a[1, 1, 1, :])
    nz = length(a[1, 1, :, 1])
    ny = length(a[1, :, 1, 1])
    nx = length(a[:, 1, 1, 1])
    a_avg_xy = Statistics.mean(a, dims=(1, 2))
    a_prime = a .- a_avg_xy 
    a_prime2 = a_prime.^2
    aprime2_norm = Array{Float64}(undef, nz, nt)
    aprime2_norm = Statistics.mean(a_prime2, dims=(1, 2)) / (u_f^2)
    return aprime2_norm
end

fld_file="langmuir_turbulence_fields.jld2"
averages_file="langmuir_turbulence_averages.jld2"
f = jldopen(fld_file)

#time and IC data 
u★ = f["IC"]["friction_velocity"]
t_save = collect(keys(f["timeseries"]["t"]))
close(f)

w = FieldTimeSeries(fld_file, "w")
u = FieldTimeSeries(fld_file, "u")
T = FieldTimeSeries(averages_file, "T")
U = FieldTimeSeries(averages_file, "U")
V = FieldTimeSeries(averages_file, "V")
wu = FieldTimeSeries(averages_file, "wu")
wv = FieldTimeSeries(averages_file, "wv")

grid = u.grid
Lx = u.grid.Lx
Ly = u.grid.Ly
Lz = u.grid.Lz
Nx = u.grid.Nx
Ny = u.grid.Ny
Nz = u.grid.Nz
Nt = length(u.times)
times = u.times

B = FieldTimeSeries{Nothing, Nothing, Center}(grid, times)
B .= g_Earth * p.β * (T.data .- p.T0)
# function calls
println("Calculating VKE")
wprime2 = VKE(w.data, u★)
initial_data = wprime2[-2, -2, :, 1] #negative indices because of the halo
wprime2_obs = Observable(initial_data)

# plotting results
n = Observable(1)
pt = 1
axis_kwargs = (xlabel="y (m)",
            ylabel="z (m)",
            aspect = AxisAspect(grid.Lx/grid.Lz),
            limits = ((0, grid.Lx), (-grid.Lz, 0)))
fig = Figure(size = (850, 850))

# w surface plane slice
wxy_title = @lift string("w(x, y, t), at z=-8 m and t = ", prettytime(times[$n]))
ax_wxy = Axis(fig[1, 1:2];
            xlabel = "x (m)",
            ylabel = "y (m)",
            aspect = DataAspect(),
            limits = ((0, grid.Lx), (0, grid.Ly)),
            title = wxy_title)
k = searchsortedfirst(znodes(grid, Face(); with_halos=false), -8)
wxyₙ = @lift view(w[$n], :, :, k)
wlims = (-0.02, 0.02)
hm_wxy = heatmap!(ax_wxy, wxyₙ;
                colorrange = wlims,
                colormap = :balance)
Colorbar(fig[1, 3], hm_wxy; label = "m s⁻¹")

# w yz plane slice
wxz_title = @lift string("w(x, z, t), at x=0 m and t = ", prettytime(times[$n]))
ax_wxz = Axis(fig[2, 1:2]; title = wxz_title, axis_kwargs...)
wxzₙ = @lift view(w[$n], 1, :, :)
hm_wxz = heatmap!(ax_wxz, wxzₙ;
                colorrange = wlims,
                colormap = :balance)

Colorbar(fig[2, 3], hm_wxz; label = "m s⁻¹")

# u xz plane slice
uxz_title = @lift string("u(x, z, t), at x=0 m and t = ", prettytime(times[$n]))
ax_uxz = Axis(fig[3, 1:2]; title = uxz_title, axis_kwargs...)
uₙ = @lift u[$n]
uxzₙ = @lift view(u[$n], 1, :, :)
ulims = (-0.1, 0.1)
ax_uxz = heatmap!(ax_uxz, uxzₙ;
                colorrange = ulims,
                colormap = :balance)

Colorbar(fig[3, 3], ax_uxz; label = "m s⁻¹")

# buoyancy with depth
ax_B = Axis(fig[1, 4:5];
            xlabel = "Buoyancy (m s⁻²)",
            ylabel = "z (m)",
            limits = ((minimum(B.data[:, :, :, :]), maximum(B.data[:, :, :, :])), nothing))
Bₙ = @lift view(B[$n], 1, 1, :)
lines!(ax_B, Bₙ)

# mean horizontal velocities with depth
ax_U = Axis(fig[2, 4:5];
            xlabel = "Velocities (m s⁻¹)",
            ylabel = "z (m)",
            limits = ((minimum(U.data[:, :, :, :]), maximum(U.data[:, :, :, :])), nothing))
Uₙ = @lift view(U[$n], 1, 1, :)
Vₙ = @lift view(V[$n], 1, 1, :)
lines!(ax_U, Uₙ; label = L"\bar{u}")
lines!(ax_U, Vₙ; label = L"\bar{v}")
axislegend(ax_U; position = :rb)

# momentum fluxes with depth
ax_fluxes = Axis(fig[3, 4:5];
                xlabel = "Momentum fluxes (m² s⁻²)",
                ylabel = "z (m)",
                limits = ((minimum(wu.data[:, :, :, :]), maximum(wu.data[:, :, :, :])), nothing))
wuₙ = @lift view(wu[$n], 1, 1, :)
wvₙ = @lift view(wv[$n], 1, 1, :)
lines!(ax_fluxes, wuₙ; label = L"\overline{wu}")
lines!(ax_fluxes, wvₙ; label = L"\overline{wv}")
axislegend(ax_fluxes; position = :rb)

#VKE
ax_VKE = Axis(fig[4, 4:5];
                xlabel = L"\overline{w'²} / u★²",
                ylabel = "z (m)",
                limits = ((0.0, 5.0), nothing))
lines!(ax_VKE, wprime2_obs, grid.z.cᵃᵃᶠ; label = L"\overline{w'²} / u★²")
axislegend(ax_VKE; position = :rb)

fig

frames = 1:length(times)

record(fig, "langmuir_turbulence_temp_buoy.mp4", frames, framerate=8) do i
    n[] = i
    wprime2_obs[] = wprime2[-2, -2, :, i]
end
