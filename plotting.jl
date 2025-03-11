using Pkg
using CairoMakie
using Oceananigans
using Oceananigans.Units: minute, minutes, hours

function plot()
    # running locally: using Pkg; Pkg.add("Oceananigans"); Pkg.add("CairoMakie"); Pkg.add("JLD2")
    Nranks = 4

    fld_file="outputs/langmuir_turbulence_fields_0_rank0.jld2"
    averages_file="outputs/langmuir_turbulence_averages_0_rank0.jld2"

    w_temp = FieldTimeSeries(fld_file, "w")
    u_temp = FieldTimeSeries(fld_file, "u")
    B_temp = FieldTimeSeries(averages_file, "B")
    U_temp = FieldTimeSeries(averages_file, "U")
    V_temp = FieldTimeSeries(averages_file, "V")
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

    println(Nx)

    w_data = Array{Float64}(undef, (Nx, Ny, Nz + 1, Nt)) #because face value
    u_data = Array{Float64}(undef, (Nx, Ny, Nz, Nt))
    B_data = Array{Float64}(undef, (1, 1, Nz, Nt))
    U_data = Array{Float64}(undef, (1, 1, Nz, Nt))
    V_data = Array{Float64}(undef, (1, 1, Nz, Nt))
    wu_data = Array{Float64}(undef, (1, 1, Nz + 1, Nt))  
    wv_data = Array{Float64}(undef, (1, 1, Nz + 1, Nt))
    B_data .= 0
    U_data .= 0
    V_data .= 0
    wu_data .= 0
    wv_data .= 0

    p = 1
    w_data[p:u_temp.grid.Nx, :, :, :] .= w_temp.data
    u_data[p:u_temp.grid.Nx, :, :, :] .= u_temp.data
    B_data .= B_data .+ B_temp.data
    U_data .= U_data .+ U_temp.data
    V_data .= V_data .+ V_temp.data
    wu_data .= wu_data .+ wu_temp.data
    wv_data .= wv_data .+ wv_temp.data

    for i in 1:Nranks-1

        p = p + u_temp.grid.Nx

        println("Loading rank $i")

        fld_file="outputs/langmuir_turbulence_fields_$(i)_rank$(i).jld2"
        averages_file="outputs/langmuir_turbulence_averages_$(i)_rank$(i).jld2"

        w_temp = FieldTimeSeries(fld_file, "w")
        u_temp = FieldTimeSeries(fld_file, "u")
        B_temp = FieldTimeSeries(averages_file, "B")
        U_temp = FieldTimeSeries(averages_file, "U")
        V_temp = FieldTimeSeries(averages_file, "V")
        wu_temp = FieldTimeSeries(averages_file, "wu")
        wv_temp = FieldTimeSeries(averages_file, "wv")
        
        w_data[p:p + w_temp.grid.Nx - 1, :, :, :] .= w_temp.data
        u_data[p:p + u_temp.grid.Nx - 1, :, :, :] .= u_temp.data
        B_data .= B_data .+ B_temp.data
        U_data .= U_data .+ U_temp.data
        V_data .= V_data .+ V_temp.data
        wu_data .= wu_data .+ wu_temp.data
        wv_data .= wv_data .+ wv_temp.data
        
    end

    B_data = B_data ./ Nranks
    U_data = U_data ./ Nranks
    V_data = V_data ./ Nranks
    wu_data = wu_data ./ Nranks
    wv_data = wv_data ./ Nranks
    #putting everything back into FieldTimeSeries
    w = FieldTimeSeries{Center, Center, Face}(grid, times)
    u = FieldTimeSeries{Face, Center, Center}(grid, times)
    B = FieldTimeSeries{Center, Center, Center}(grid, times)
    U = FieldTimeSeries{Center, Center, Center}(grid, times)
    V = FieldTimeSeries{Center, Center, Center}(grid, times)
    wu = FieldTimeSeries{Center, Center, Face}(grid, times)
    wv = FieldTimeSeries{Center, Center, Face}(grid, times)

    w .= w_data
    u .= u_data
    B .= B_data
    U .= U_data
    V .= V_data
    wu .= wu_data
    wv .= wv_data

    #begin plotting
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

end
