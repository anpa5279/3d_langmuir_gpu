using Pkg
using Statistics
using CairoMakie
using Printf
using JLD2
using Oceananigans
using Oceananigans.Units: minute, minutes, hours
using Oceananigans.BuoyancyFormulations: g_Earth
mutable struct Params
    Nx::Int         # number of points in each of x direction
    Ny::Int         # number of points in each of y direction
    Nz::Int         # number of points in the vertical direction
    Lx::Float64     # (m) domain horizontal extents
    Ly::Float64     # (m) domain horizontal extents
    Lz::Float64     # (m) domain depth 
    N²::Float64     # s⁻², initial and bottom buoyancy gradient
    initial_mixed_layer_depth::Float64 # m 
    Q::Float64      # W m⁻², surface heat flux. cooling is positive
    cᴾ::Float64     # J kg⁻¹ K⁻¹, specific heat capacity of seawater
    ρₒ::Float64     # kg m⁻³, average density at the surface of the world ocean
    dTdz::Float64   # K m⁻¹, temperature gradient
    T0::Float64     # C, temperature at the surface   
    β::Float64      # 1/K, thermal expansion coefficient
    u₁₀::Float64    # (m s⁻¹) wind speed at 10 meters above the ocean
    La_t::Float64   # Langmuir turbulence number
end

#defaults, these can be changed directly below 128, 128, 160, 320.0, 320.0, 96.0
p = Params(128, 128, 160, 320.0, 320.0, 96.0, 5.3e-9, 33.0, 5.0, 3991.0, 1000.0, 0.01, 17.0, 2.0e-4, 5.75, 0.29)
function VKE(a, u_f)
    a= copy(parent(a))
    nt = length(a[1, 1, 1, :])
    nz = length(a[1, 1, :, 1])
    ny = length(a[1, :, 1, 1])
    nx = length(a[:, 1, 1, 1])
    a_avg_xy = Statistics.mean(a, dims=(1, 2))
    a_avg_xy = repeat(a_avg_xy, nx, ny, 1)
    a_prime = a_avg_xy .- a
    a_prime2 = a_prime.^2
    aprime2_norm = Array{Float64}(undef, nz, nt)
    aprime2_norm = Statistics.mean(a_prime2, dims=(1, 2)) / (u_f^2)
    return aprime2_norm
end

function plot()
    Nranks = 4

    fld_file="outputs/langmuir_turbulence_fields_0.jld2"
    averages_file="outputs/langmuir_turbulence_averages_0.jld2"

    # Load the data
    f = jldopen(fld_file)
    fa = jldopen(averages_file)
    # required IC from model
    global u★ = f["IC"]["friction_velocity"]
    u_stokes = f["IC"]["stokes_velocity"]
    u₁₀ = f["IC"]["wind_speed"]
    #loading in the data
    w_temp = f["timeseries"]["w"]
    u_temp = f["timeseries"]["u"]
    b_temp = f["timeseries"]["b"]
    t_temp = keys(f["timeseries"]["t"])
    U_temp = FieldTimeSeries(averages_file, "U")
    V_temp = FieldTimeSeries(averages_file, "V")
    wu_temp = FieldTimeSeries(averages_file, "wu")
    wv_temp = FieldTimeSeries(averages_file, "wv")

    Nt = length(t_temp)
    grid = RectilinearGrid(size = (p.Nx, p.Ny, p.Nz), extent = (p.Lx, p.Ly, p.Lz), halo = (3, 3, 3))

    times = Array{Float64}(undef, Nt)
    w_data = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz + 1, Nt)) #because face value
    u_data = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz, Nt))
    b_data = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz, Nt))
    U_data = Array{Float64}(undef, (1, 1, p.Nz, Nt))
    V_data = Array{Float64}(undef, (1, 1, p.Nz, Nt))
    wu_data = Array{Float64}(undef, (1, 1, p.Nz, Nt))  
    wv_data = Array{Float64}(undef, (1, 1, p.Nz, Nt))
    U_data .= 0
    V_data .= 0
    wu_data .= 0
    wv_data .= 0

    nn = 1 
    Nr = Int(p.Nx / Nranks)
    for i in 1:Nt
        step = t_temp[i]
        times[i] = f["timeseries"]["t"][step]
        w_data[nn:nn + Nr - 1, :, :, :] .= w_temp[step][grid.Hx + 1:grid.Hx + Nr, grid.Hy + 1:grid.Hy + p.Ny, grid.Hz + 1:grid.Hz + p.Nz + 1]
        u_data[nn:nn + Nr - 1, :, :, :] .= u_temp[step][grid.Hx + 1:grid.Hx + Nr, grid.Hy + 1:grid.Hy + p.Ny, grid.Hz + 1:grid.Hz + p.Nz]
        b_data[nn:nn + Nr - 1, :, :, :] .= b_temp[step][grid.Hx + 1:grid.Hx + Nr, grid.Hy + 1:grid.Hy + p.Ny, grid.Hz + 1:grid.Hz + p.Nz]
        @show i
    end

    U_data .= U_data .+ U_temp.data[:, :, 1:p.Nz, :]
    V_data .= V_data .+ V_temp.data[:, :, 1:p.Nz, :]
    wu_data .= wu_data .+ wu_temp.data[:, :, 1:p.Nz, :]
    wv_data .= wv_data .+ wv_temp.data[:, :, 1:p.Nz, :]

    for i in 1:Nranks-1

        nn = nn + Nr

        println("Loading rank $i")

        fld_file="outputs/langmuir_turbulence_fields_$(i).jld2"
        averages_file="outputs/langmuir_turbulence_averages_$(i).jld2"

        f = jldopen(fld_file)
        fa = jldopen(averages_file)

        w_temp = f["timeseries"]["w"]
        u_temp = f["timeseries"]["u"]
        b_temp = f["timeseries"]["b"]
        U_temp = FieldTimeSeries(averages_file, "U")
        V_temp = FieldTimeSeries(averages_file, "V")
        W_temp = FieldTimeSeries(averages_file, "W")
        wu_temp = FieldTimeSeries(averages_file, "wu")
        wv_temp = FieldTimeSeries(averages_file, "wv")
        
        for k in 1:Nt
            @show k
            step = t_temp[k]
            times[k] = f["timeseries"]["t"][step]
            w_data[nn:nn + Nr - 1, :, :, :] .= w_temp[step][grid.Hx + 1:grid.Hx + Nr, grid.Hy + 1:grid.Hy + p.Ny, grid.Hz + 1:grid.Hz + p.Nz + 1]
            u_data[nn:nn + Nr - 1, :, :, :] .= u_temp[step][grid.Hx + 1:grid.Hx + Nr, grid.Hy + 1:grid.Hy + p.Ny, grid.Hz + 1:grid.Hz + p.Nz]
            b_data[nn:nn + Nr - 1, :, :, :] .= b_temp[step][grid.Hx + 1:grid.Hx + Nr, grid.Hy + 1:grid.Hy + p.Ny, grid.Hz + 1:grid.Hz + p.Nz]
        end
        U_data .= U_data .+ U_temp.data[:, :, 1:p.Nz, :]
        V_data .= V_data .+ V_temp.data[:, :, 1:p.Nz, :]
        wu_data .= wu_data .+ wu_temp.data[:, :, 1:p.Nz, :]
        wv_data .= wv_data .+ wv_temp.data[:, :, 1:p.Nz, :]
        
    end

    #averaging
    B_avg = b_data ./ (p.Nx * p.Ny * p.Nz)
    U_data = U_data ./ Nranks
    V_data = V_data ./ Nranks
    wu_data = wu_data ./ Nranks
    wv_data = wv_data ./ Nranks

    #putting everything back into FieldTimeSeries
    w = FieldTimeSeries{Center, Center, Face}(grid, times)
    u = FieldTimeSeries{Face, Center, Center}(grid, times)
    b = FieldTimeSeries{Center, Center, Center}(grid, times)
    B = FieldTimeSeries{Center, Center, Center}(grid, times)
    U = FieldTimeSeries{Center, Center, Center}(grid, times)
    V = FieldTimeSeries{Center, Center, Center}(grid, times)
    wu = FieldTimeSeries{Center, Center, Face}(grid, times)
    wv = FieldTimeSeries{Center, Center, Face}(grid, times)

    global w .= w_data
    global u .= u_data
    global b .= b_data
    global B .= B_avg
    global U .= U_data
    global V .= V_data
    global wu .= wu_data
    global wv .= wv_data

    # function calls
    wprime2 = VKE(w.data, u★)
    @show size(wprime2)
    initial_data = wprime2[1, 1, :, 1]
    x_obs = Observable(initial_data)

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

    # u yz plane slice
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
    ax_fluxes = Axis(fig[4, 4:5];
                    xlabel = L"\overline{w'²} / u★²",
                    ylabel = "z (m)",
                    limits = ((0.0, 5.0), nothing))
    lines!(ax_fluxes, x_obs,  w.grid.z.cᵃᵃᶜ[1:p.Nz+1]; label = L"\overline{w'²} / u★²")
    axislegend(ax_fluxes; position = :rb)

    fig

    frames = 1:length(times)

    record(fig, "plotting.mp4", frames, framerate=8) do i
        n[] = i
        x_obs[] = wprime2[:, i]
    end 
end 

function only_plot()
    grid = RectilinearGrid(size = (p.Nx, p.Ny, p.Nz), extent = (p.Lx, p.Ly, p.Lz), halo = (3, 3, 3))
    # function calls
    wprime2 = VKE(w.data, u★)
    @show size(wprime2)
    initial_data = wprime2[1, 1, :, 1]
    x_obs = Observable(initial_data)

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

    # u yz plane slice
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
    ax_fluxes = Axis(fig[4, 4:5];
                    xlabel = L"\overline{w'²} / u★²",
                    ylabel = "z (m)",
                    limits = ((0.0, 5.0), nothing))
    lines!(ax_fluxes, x_obs,  w.grid.z.cᵃᵃᶜ[1:p.Nz+1]; label = L"\overline{w'²} / u★²")
    axislegend(ax_fluxes; position = :rb)

    fig

    frames = 1:length(times)

    record(fig, "plotting.mp4", frames, framerate=8) do i
        n[] = i
        x_obs[] = wprime2[:, i]
    end 
end 