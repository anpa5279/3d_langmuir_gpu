using Pkg
using Statistics
using CairoMakie
using Printf
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
p = Params(32, 32, 32, 128, 128, 96.0, 5.3*(10^(-9)), 33.0, 5.0, 3991.0, 1000.0, 0.006667, 17.0, 2.0e-4, 5.75, 0.29)

#functions
function plot()
    Nranks = 4

    fld_file="outputs/langmuir_turbulence_fields_0.jld2"
    averages_file="outputs/langmuir_turbulence_averages_0.jld2"

    w_temp = FieldTimeSeries(fld_file, "w")
    u_temp = FieldTimeSeries(fld_file, "u")
    B_temp = FieldTimeSeries(fld_file, "b")
    U_temp = FieldTimeSeries(averages_file, "U")
    V_temp = FieldTimeSeries(averages_file, "V")
    W_temp = FieldTimeSeries(averages_file, "W")
    wu_temp = FieldTimeSeries(averages_file, "wu")
    wv_temp = FieldTimeSeries(averages_file, "wv")

    Lx = Nranks * u_temp.grid.Lx
    Ly = u_temp.grid.Ly
    Lz = u_temp.grid.Lz
    Nx = Nranks * u_temp.grid.Nx
    Ny = u_temp.grid.Ny
    Nz = u_temp.grid.Nz
    Nt = length(u_temp.times)
    grid = RectilinearGrid(size = (Nx, Ny, Nz), extent = (Lx, Ly, Lz))
    times = u_temp.times

    w_data = Array{Float64}(undef, (Nx, Ny, Nz + 1, Nt)) #because face value
    u_data = Array{Float64}(undef, (Nx, Ny, Nz, Nt))
    b_data = Array{Float64}(undef, (Nx, Ny, Nz, Nt))
    U_data = Array{Float64}(undef, (1, 1, Nz, Nt))
    V_data = Array{Float64}(undef, (1, 1, Nz, Nt))
    W_data = Array{Float64}(undef, (1, 1, Nz, Nt))
    wu_data = Array{Float64}(undef, (1, 1, Nz + 1, Nt))  
    wv_data = Array{Float64}(undef, (1, 1, Nz + 1, Nt))
    U_data .= 0
    V_data .= 0
    W_data .= 0
    wu_data .= 0
    wv_data .= 0

    p = 1
    w_data[p:p + w_temp.grid.Nx - 1, :, :, :] .= w_temp.data
    u_data[p:p + u_temp.grid.Nx - 1, :, :, :] .= u_temp.data
    b_data[p:p + b.grid.Nx - 1, :, :, :] .= b.data
    U_data .= U_data .+ U_temp.data
    V_data .= V_data .+ V_temp.data
    W_data .= W_data .+ W_temp.data
    wu_data .= wu_data .+ wu_temp.data
    wv_data .= wv_data .+ wv_temp.data

    for i in 1:Nranks-1

        p = p + u_temp.grid.Nx

        println("Loading rank $i")

        fld_file="outputs/langmuir_turbulence_fields_$(i).jld2"
        averages_file="outputs/langmuir_turbulence_averages_$(i).jld2"

        w_temp = FieldTimeSeries(fld_file, "w")
        u_temp = FieldTimeSeries(fld_file, "u")
        B_temp = FieldTimeSeries(fld_file, "b")
        U_temp = FieldTimeSeries(averages_file, "U")
        V_temp = FieldTimeSeries(averages_file, "V")
        W_temp = FieldTimeSeries(averages_file, "W")
        wu_temp = FieldTimeSeries(averages_file, "wu")
        wv_temp = FieldTimeSeries(averages_file, "wv")
        
        w_data[p:p + w_temp.grid.Nx - 1, :, :, :] .= w_temp.data
        u_data[p:p + u_temp.grid.Nx - 1, :, :, :] .= u_temp.data
        b_data[p:p + w_temp.grid.Nx - 1, :, :, :] .= B_temp.data
        U_data .= U_data .+ U_temp.data
        V_data .= V_data .+ V_temp.data
        W_data .= W_data .+ W_temp.data
        wu_data .= wu_data .+ wu_temp.data
        wv_data .= wv_data .+ wv_temp.data
        
    end

    #averaging
    B_avg = b_data ./ (Nx * Ny * Nz)
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

    w .= w_data
    u .= u_data
    T .= T_data
    b .= b_data
    B .= B_avg
    U .= U_data
    V .= V_data
    wu .= wu_data
    wv .= wv_data

    #begin plotting
    n = Observable(1)
    axis_kwargs = (xlabel="x (m)",
                ylabel="z (m)",
                aspect = AxisAspect(p.Lx/p.Lz),
                limits = ((0, p.Lx), (-p.Lz, 0)))
    fig = Figure(size = (850, 1150))
    title = @lift @sprintf("t = %s", prettytime(times[$n]))

    # w surface plane slice
    wxy_title = @lift string("w(x, y, t), t = ", prettytime(times[$n]))
    ax_wxy = Axis(fig[1, 1:2];
                xlabel = "x (m)",
                ylabel = "y (m)",
                aspect = DataAspect(),
                limits = ((0, p.Lx), (0, p.Ly)),
                title = wxy_title)
    k = searchsortedfirst(znodes(grid, Face(); with_halos=false), -2)
    wxyₙ = @lift view(w[$n], :, :, k)
    wlims = (-0.02, 0.02)
    hm_wxy = heatmap!(ax_wxy, wxyₙ;
                    colorrange = wlims,
                    colormap = :balance)
    Colorbar(fig[1, 3], hm_wxy; label = "m s⁻¹")

    # w xz plane slice
    wxz_title = @lift string("w(x, z, t), t = ", prettytime(times[$n]))
    ax_wxz = Axis(fig[2, 1:2]; title = wxz_title, axis_kwargs...)
    wxzₙ = @lift view(w[$n], :, p.Ny/2, :)
    hm_wxz = heatmap!(ax_wxz, wxzₙ;
                    colorrange = wlims,
                    colormap = :balance)

    Colorbar(fig[2, 3], hm_wxz; label = "m s⁻¹")

    # w yz plane slice
    wyz_title = @lift string("w(y, z, t), t = ", prettytime(times[$n]))
    ax_wxz = Axis(fig[3, 1:2]; title = wyz_title, xlabel = "y (m)",
                ylabel = "z (m)",
                aspect = DataAspect(),
                limits = ((0, p.Ly), (-p.Lz, 0)))
    wxzₙ = @lift view(w[$n], p.Nx/2, :, :)
    hm_wxz = heatmap!(ax_wxz, wxzₙ;
                    colorrange = wlims,
                    colormap = :balance)

    Colorbar(fig[3, 3], hm_wxz; label = "m s⁻¹")

    # u xz plane slice
    uxz_title = @lift string("u(x, z, t), t = ", prettytime(times[$n]))
    ax_uxz = Axis(fig[4, 1:2]; title = uxz_title, axis_kwargs...)
    uₙ = @lift u[$n]
    uxzₙ = @lift view(u[$n], :, p.Ny/2, :)
    ulims = (-0.06, 0.06)
    ax_uxz = heatmap!(ax_uxz, uxzₙ;
                    colorrange = ulims,
                    colormap = :balance)

    Colorbar(fig[4, 3], ax_uxz; label = "m s⁻¹")

    # temperature vertical plane slice
    xT, yT, zT = nodes(T)
    T_title = @lift string("T(x, z, t), t = ", prettytime(times[$n]))   
    ax_T  = Axis(fig[4, 4:5]; title = T_title, xlabel = "y (m)",
                ylabel = "z (m)",
                aspect = DataAspect(),
                limits = ((0, p.Ly), (-p.Lz, 0)))
    Tₙ = @lift interior(T[$n],  1, :, :)
    Tlims = (16.5, 17.0)
    hm_T = heatmap!(ax_T, xT, zT, Tₙ; colormap = :thermal, colorrange = Tlims)
    Colorbar(fig[4, 6], hm_T; label = "ᵒC")

    # buoyancy with depth
    ax_B = Axis(fig[1, 4:5];
    xlabel = "Buoyancy (m s⁻²)",
    ylabel = "z (m)")
    Bₙ = @lift view(B[$n], 1, 1, :)
    lines!(ax_B, Bₙ)

    # mean horizontal velocities with depth
    ax_U = Axis(fig[2, 4:5];
    xlabel = "Velocities (m s⁻¹)",
    ylabel = "z (m)",
    limits = ((-0.07, 0.07), nothing))
    Uₙ = @lift view(U[$n], 1, 1, :)
    Vₙ = @lift view(V[$n], 1, 1, :)
    lines!(ax_U, Uₙ; label = L"\bar{u}")
    lines!(ax_U, Vₙ; label = L"\bar{v}")
    axislegend(ax_U; position = :rb)

    # momentum fluxes with depth
    ax_fluxes = Axis(fig[3, 4:5];
    xlabel = "Momentum fluxes (m² s⁻²)",
    ylabel = "z (m)",
    limits = ((-3.5e-5, 3.5e-5), nothing))
    wuₙ = @lift view(wu[$n], 1, 1, :)
    wvₙ = @lift view(wv[$n], 1, 1, :)
    lines!(ax_fluxes, wuₙ; label = L"mean $wu$")
    lines!(ax_fluxes, wvₙ; label = L"mean $wv$")
    axislegend(ax_fluxes; position = :rb)

    fig

    frames = 1:length(times)

    record(fig, "langmuir_turbulence_comparison.mp4", frames, framerate=8) do i
        n[] = i
    end
end 

function fluctuation(a, a_avg)
    a_fluct = Array{Float64}(undef, size(a_avg))
    for i in 1:size(a, 1)
        for j in 1:size(a, 2)
            for k in 1:size(a, 3)
                a_fluct[i, j, k] = a[i, j, k] - a_avg[k]
            end
        end
    end
    return a_fluct
end 
function fluct_squared(a, a_avg)
    a_fluct = Array{Float64}(undef, size(a_avg))
    for i in 1:size(a, 1)
        for j in 1:size(a, 2)
            for k in 1:size(a, 3)
                a_fluct2[i, j, k] = (a[i, j, k] - a_avg[k])^2
            end
        end
    end
    return a_fluct2
end
