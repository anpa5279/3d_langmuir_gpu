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

function plot_ncar()
    # collecting NCAR LES data
    dtn = cd(readdir, "data")
    nvar = 5
    Nt = Int(length(dtn)/2)
    u_ncar = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz, Nt))
    v_ncar = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz, Nt))
    w_ncar = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz, Nt))
    T_ncar = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz, Nt))
    var = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz, nvar))
    j = 1

    for i in 1:length(dtn)
        if contains(dtn[i], ".con")==0 && contains(dtn[i], "u.mp.")
            tmp = read("data/"* dtn[i])
            #@show "data/"* dtn[i]
            var=reshape(reinterpret(Float64, tmp), p.Nx, p.Ny, p.Nz, nvar)
            u_ncar[:,:,:,j] = var[:,:,:,1]
            v_ncar[:,:,:,j] = var[:,:,:,2]
            w_ncar[:,:,:,j] = var[:,:,:,3]
            T_ncar[:,:,:,j] = var[:,:,:,4]
            j = j + 1
            #@show j
        end
    end
    #ccalculating plotting variables
    b_ncar = g_Earth * p.β * (T_ncar .- p.T0)
    b_ncar_avg = Statistics.mean(b_ncar, dims=(1, 2))
    u_ncar_avg = Statistics.mean(u_ncar, dims=(1, 2))
    v_ncar_avg = Statistics.mean(v_ncar, dims=(1, 2))
    wu_ncar = Statistics.mean(w_ncar .* u_ncar, dims=(1, 2))
    wv_ncar = Statistics.mean(w_ncar .* v_ncar, dims=(1, 2))

    w_ncarles = FieldTimeSeries{Center, Center, Face}(grid, times)
    u_ncarles = FieldTimeSeries{Face, Center, Center}(grid, times)
    b_ncarles = FieldTimeSeries{Center, Center, Center}(grid, times)
    B_ncarles = FieldTimeSeries{Center, Center, Center}(grid, times)
    U_ncarles = FieldTimeSeries{Center, Center, Center}(grid, times)
    V_ncarles = FieldTimeSeries{Center, Center, Center}(grid, times)
    wu_ncarles = FieldTimeSeries{Center, Center, Face}(grid, times)
    wv_ncarles = FieldTimeSeries{Center, Center, Face}(grid, times)

    global w_ncarles .= w_ncar
    global u_ncarles .= u_ncar
    global b_ncarles .= b_ncar
    global B_ncarles .= b_ncar_avg
    global U_ncarles .= u_ncar_avg
    global V_ncarles .= v_ncar_avg
    global wu_ncarles .= wu_ncar
    global wv_ncarles .= wv_ncar

    # manipulating NCAR data
    wprime2_ncarles = VKE(w_ncar, 0.5301e-02)
    b_ncarles = g_Earth * p.β * (T_ncar)

    initial_data = wprime2_ncarles[1, 1, :, 1]
    wprime2_obs = Observable(initial_data)

    # plotting results
    n = Observable(1)
    pt = 1
    axis_kwargs = (xlabel="y (m)",
                ylabel="z (m)",
                aspect = AxisAspect(p.Lx/p.Lz),
                limits = ((0, p.Lx), (-p.Lz, 0)))
    fig = Figure(size = (850, 850))

    # w surface plane slice
    wxy_title = @lift string("w(x, y, t), at z=-8 m and t = ", prettytime(times[$n]))
    ax_wxy = Axis(fig[1, 1:2];
                xlabel = "x (m)",
                ylabel = "y (m)",
                aspect = DataAspect(),
                limits = ((0, grid.Lx), (0, grid.Ly)),
                title = wxy_title)
    k = searchsortedfirst(znodes(grid, Face(); with_halos=true), -8)
    wxyₙ = @lift view(w_ncarles[$n], :, :, k)
    wlims = (-0.02, 0.02)
    hm_wxy = heatmap!(ax_wxy, wxyₙ;
                    colorrange = wlims,
                    colormap = :balance)
    Colorbar(fig[1, 3], hm_wxy; label = "m s⁻¹")

    # w yz plane slice
    wxz_title = @lift string("w(x, z, t), at x=0 m and t = ", prettytime(times[$n]))
    ax_wxz = Axis(fig[2, 1:2]; title = wxz_title, axis_kwargs...)
    wxzₙ = @lift view(w_ncarles[$n], 1, :, :)
    hm_wxz = heatmap!(ax_wxz, wxzₙ;
                    colorrange = wlims,
                    colormap = :balance)

    Colorbar(fig[2, 3], hm_wxz; label = "m s⁻¹")

    # u yz plane slice
    uxz_title = @lift string("u(x, z, t), at x=0 m and t = ", prettytime(times[$n]))
    ax_uxz = Axis(fig[3, 1:2]; title = uxz_title, axis_kwargs...)
    uₙ = @lift u_ncarles[$n]
    uxzₙ = @lift view(u_ncarles[$n], 1, :, :)
    ulims = (-0.1, 0.1)
    ax_uxz = heatmap!(ax_uxz, uxzₙ;
                    colorrange = ulims,
                    colormap = :balance)

    Colorbar(fig[3, 3], ax_uxz; label = "m s⁻¹")

    # buoyancy with depth
    ax_B = Axis(fig[1, 4:5];
                xlabel = "Buoyancy (m s⁻²)",
                ylabel = "z (m)",
                limits = ((minimum(B_ncarles.data[:, :, :, :]), maximum(B_ncarles.data[:, :, :, :])), nothing))
    Bₙ = @lift view(B_ncarles[$n], 1, 1, :)
    lines!(ax_B, Bₙ)

    # mean horizontal velocities with depth
    ax_U = Axis(fig[2, 4:5];
                xlabel = "Velocities (m s⁻¹)",
                ylabel = "z (m)",
                limits = ((minimum(U_ncarles.data[:, :, :, :]), maximum(U_ncarles.data[:, :, :, :])), nothing))
    Uₙ = @lift view(U_ncarles[$n], 1, 1, :)
    Vₙ = @lift view(V_ncarles[$n], 1, 1, :)
    lines!(ax_U, Uₙ; label = L"\bar{u}")
    lines!(ax_U, Vₙ; label = L"\bar{v}")
    axislegend(ax_U; position = :rb)

    # momentum fluxes with depth
    ax_fluxes = Axis(fig[3, 4:5];
                    xlabel = "Momentum fluxes (m² s⁻²)",
                    ylabel = "z (m)",
                    limits = ((minimum(wu_ncarles.data[:, :, :, :]), maximum(wu_ncarles.data[:, :, :, :])), nothing))
    wuₙ = @lift view(wu_ncarles[$n], 1, 1, :)
    wvₙ = @lift view(wv_ncarles[$n], 1, 1, :)
    lines!(ax_fluxes, wuₙ; label = L"\overline{wu}")
    lines!(ax_fluxes, wvₙ; label = L"\overline{wv}")
    axislegend(ax_fluxes; position = :rb)

    #VKE
    ax_fluxes = Axis(fig[4, 4:5];
                    xlabel = L"\overline{w'²} / u★²",
                    ylabel = "z (m)",
                    limits = ((0.0, 5.0), nothing))
    lines!(ax_fluxes, wprime2_obs,  w.grid.z.cᵃᵃᶜ[1:p.Nz+1]; label = L"\overline{w'²} / u★²")
    axislegend(ax_fluxes; position = :rb)

    fig

    frames = 1:length(times)

    record(fig, "plotting_ncarles.mp4", frames, framerate=8) do i
        n[] = i
        wprime2_obs[] = wprime2_ncarles[:, i]
    end 
end