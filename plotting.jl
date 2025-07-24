using Pkg
using OffsetArrays
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
p = Params(128, 128, 160, 320.0, 320.0, 96.0, 5.3e-9, 33.0, 0.0, 4200.0, 1000.0, 0.01, 17.0, 2.0e-4, 5.75, 0.3)
grid = RectilinearGrid(size = (p.Nx, p.Ny, p.Nz), extent = (p.Lx, p.Ly, p.Lz), halo = (3, 3, 3))

include("stokes.jl")

function make_field_center(data, times)
    field = FieldTimeSeries{Center, Center, Center}(grid, times)
    field.data .= data
    return field
end

function make_field_face(data, times)
    # Create a FieldTimeSeries object from the data
    field = FieldTimeSeries{Center, Center, Face}(grid, times)
    field.data .= data
    return field
end

function video()
    Nranks = 4
    fld_file="outputs/langmuir_turbulence_fields_0.jld2"
    fld_file="localoutputs/NBP_fields.jld2"
    fld_file="localoutputs/NBP_averages.jld2"
    f = jldopen(fld_file)

    #time and IC data 
    u★ = f["IC"]["friction_velocity"]
    #u_stokes = f["IC"]["stokes_velocity"]
    #u₁₀ = f["IC"]["wind_speed"]
    t_save = collect(keys(f["timeseries"]["t"]))
    close(f)

    Nt = length(t_save)

    times = Array{Float64}(undef, Int(Nt/2))
    w_data = OffsetArray{Float64}(undef, -grid.Hx+1 : p.Nx+grid.Hx,
                                        -grid.Hy+1 : p.Ny+grid.Hy,
                                        -grid.Hz+1 : p.Nz+1+grid.Hz,
                                        1 : Int(Nt/2))
    u_data = OffsetArray{Float64}(undef, -grid.Hx+1 : p.Nx+grid.Hx,
                                        -grid.Hy+1 : p.Ny+grid.Hy,
                                        -grid.Hz+1 : p.Nz+grid.Hz,
                                        1 : Int(Nt/2))
    #b_data = OffsetArray{Float64}(undef, -grid.Hx+1 : p.Nx+grid.Hx,
    #                                    -grid.Hy+1 : p.Ny+grid.Hy,
    #                                    -grid.Hz+1 : p.Nz+grid.Hz,
    #                                    1 : Nt)
    T_data =  Array{Float64}(undef, (1, 1, p.Nz, Int(Nt/2)))
    U_data = Array{Float64}(undef, (1, 1, p.Nz, Int(Nt/2)))
    V_data = Array{Float64}(undef, (1, 1, p.Nz, Int(Nt/2)))
    wu_data = Array{Float64}(undef, (1, 1, p.Nz + 1, Int(Nt/2)))
    wv_data =   Array{Float64}(undef, (1, 1, p.Nz + 1, Int(Nt/2)))
    T_data .= 0
    U_data .= 0
    V_data .= 0
    wu_data .= 0
    wv_data .= 0

    for i in 0:Nranks-1
        println("Loading rank $i")

        fld_file="outputs/langmuir_turbulence_fields_$(i).jld2"
        averages_file="outputs/langmuir_turbulence_averages_$(i).jld2"

        f = jldopen(fld_file)
        T_temp = FieldTimeSeries(averages_file, "T_avg")
        U_temp = FieldTimeSeries(averages_file, "U")
        V_temp = FieldTimeSeries(averages_file, "V")
        W_temp = FieldTimeSeries(averages_file, "W")
        wu_temp = FieldTimeSeries(averages_file, "wu")
        wv_temp = FieldTimeSeries(averages_file, "wv")
        if i == 0 #first rank
            shift = 0
            Nr = Int(p.Nx / Nranks + grid.Hx)
            xrange = 1 : Nr #last rank
        elseif i == Nranks - 1
            shift  = -2 * grid.Hx
            Nr = Int(p.Nx / Nranks + grid.Hx)
            xrange = grid.Hx + 1 : grid.Hx + Nr
        else #middle ranks
            shift = grid.Hx
            Nr = Int(p.Nx / Nranks)
            xrange = grid.Hx + 1 : grid.Hx + Nr
        end 
        nn = 1 + shift + i * Nr - grid.Hx
        w_all = [f["timeseries"]["w"][t][xrange, :, :] for t in t_save]
        u_yplane = [f["timeseries"]["u"][t][1, :, :] for t in t_save]
        #b_all = [f["timeseries"]["b"][t][xrange, :, :] for t in t_save]
        j = 1
        for k in 1:2:Nt
            @show k, j
            times[j] = f["timeseries"]["t"][t_save[j]]
            local w = w_all[k]
            local u = u_yplane[k]
            #local b = b_all[k]
            w_data[nn:nn + Nr - 1, :, :, j] = w
            u_data[1, :, :, j] = u
            #b[nn:nn + Nr - 1, :, :, k] = b
            j += 1
            #removing the data from memory
            w = nothing
            u = nothing
            #b = nothing
            GC.gc()
        end
        j = nothing
        GC.gc()
        T_data .= T_data .+ T_temp.data[:, :, 1:p.Nz, 1:Int(Nt/2)]
        U_data .= U_data .+ U_temp.data[:, :, 1:p.Nz, 1:Int(Nt/2)]
        V_data .= V_data .+ V_temp.data[:, :, 1:p.Nz, 1:Int(Nt/2)]
        wu_data .= wu_data .+ wu_temp.data[:, :, 1:p.Nz + 1, 1:Int(Nt/2)]
        wv_data .= wv_data .+ wv_temp.data[:, :, 1:p.Nz + 1, 1:Int(Nt/2)]
        #removing the data from memory
        w_all = nothing
        u_yplane = nothing
        b_all = nothing
        xrange = nothing
        Nr = nothing 
        shift = nothing
        T_temp = nothing
        U_temp = nothing
        V_temp = nothing
        wu_temp = nothing
        wv_temp = nothing
        GC.gc() 
        close(f)
    end
    t_save = nothing
    GC.gc()

    #averaging
    println("Averaging data")
    T_data = T_data ./ Nranks
    U_data = U_data ./ Nranks
    V_data = V_data ./ Nranks
    wu_data = wu_data ./ Nranks
    wv_data = wv_data ./ Nranks

    #putting everything back into FieldTimeSeries
    println("Putting data into FieldTimeSeries")
    w  = make_field_face(w_data, times)
    w_data = nothing
    GC.gc()
    u  = make_field_center(u_data, times)
    u_data = nothing
    GC.gc()
    #b = make_field_center(b_data, times)
    #b_data = nothing
    #GC.gc()
    B = FieldTimeSeries{Center, Center, Center}(grid, times)
    U = FieldTimeSeries{Center, Center, Center}(grid, times)
    V = FieldTimeSeries{Center, Center, Center}(grid, times)
    wu = FieldTimeSeries{Center, Center, Face}(grid, times)
    wv = FieldTimeSeries{Center, Center, Face}(grid, times)
    B .= g_Earth * p.β * (T_data .- p.T0)
    T_data = nothing
    GC.gc()
    U .= U_data
    U_data = nothing
    GC.gc()
    V .= V_data
    V_data = nothing
    GC.gc()
    wu .= wu_data
    wu_data = nothing
    GC.gc()
    wv .= wv_data
    wv_data = nothing
    GC.gc()
    # function calls
    println("Calculating VKE")
    w_fluct = fluctuations(w.data)
    w_fluct2 = w_fluct.^2
    wprime2 = Statistics.mean(a_prime2, dims=(1, 2)) / (u_f^2)
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
end

function stokes_plot()
    Nranks = 4
    dudz = Array{Float64}(undef, (1, 1, p.Nz))
    for i in 0:Nranks-1
        println("Loading rank $i")

        fld_file="outputs/langmuir_turbulence_fields_0.jld2"

        f = jldopen(fld_file)
        keys(f)
        dudz = f["stokes_drift_field"]
        GC.gc() 
        close(f)
    end
    z = collect(-p.Lz + grid.z.Δᵃᵃᶜ/2 : grid.z.Δᵃᵃᶜ : -grid.z.Δᵃᵃᶜ/2)
    n = Observable(1)
    fig 
    save("oceananigans_stokes_drift.png", fig)
    #plotting velocity gradient profile
    fig2 = Figure()
    ax2 = Axis(fig2[1, 1], xlabel = "duˢ/dz [1/s]", ylabel = "z [m]", title = "Stokes drift gradient")
    lines!(ax2, dudz, label = "Oceananigans Stokes drift gradient")
    fig2
    axislegend(ax2; position = :rb)
    save("oceananigans_stokes_drift_gradient.png", fig2)
end 