using Oceananigans
using Statistics
using Plots
using JLD2

function load_data(case)
    case_name = basename(case)
    field_loc = chop(case; tail = length(case_name))
    i = 0
    for (root, dirs, files) in walkdir(field_loc)
        global ranks = 0
        for file in files
            if endswith(file, ".jld2") && occursin(case_name, file) ranks += 1 end
        end
        for file in files
            if endswith(file, ".jld2") && occursin(case_name, file)
                global case_file = joinpath(root, file)
                global file_path = jldopen(case_file)
                if i == 0
                    global times = file_path["timeseries/t"]
                    global grid1 = file_path["grid"]
                    global Nx = grid1["Nx"]
                    global Ny = grid1["Ny"]
                    global Nz = grid1["Nz"]
                    global Hx = grid1["Hx"]
                    global Hy = grid1["Hy"]
                    global Hz = grid1["Hz"]
                    global Nt = length(times)
                    global u = zeros(Float64, ranks * Nx + 1, Ny, Nz, Nt)
                    global v = zeros(Float64, ranks * Nx, Ny + 1, Nz, Nt)
                    global w = zeros(Float64, ranks * Nx, Ny, Nz + 1, Nt)
                    global ν = zeros(Float64, ranks * Nx, Ny, Nz, Nt)
                    global t = zeros(Float64, Nt)
                    @show size(u), size(v), size(w), size(ν), size(t)
                end
                global u_temp = file_path["timeseries/u"]
                global v_temp = file_path["timeseries/v"]
                global w_temp = file_path["timeseries/w"]
                global ν_temp = file_path["timeseries/νₑ"]
                global start = 1 + i * Nx
                global collect_f = (Nx + 1) + i * Nx
                global collect_c = Nx + i * Nx
                global nt = 0
                @show collect_f, collect_c, start
                for tstep in keys(times)
                    global nt += 1
                    global u[start:collect_f, :, :, nt] = u_temp[tstep][Hx+1:Nx+Hx+1, Hy+1:Ny+Hy, Hz+1:Nz+Hz]
                    global v[start:collect_c, :, :, nt] = v_temp[tstep][Hx+1:Nx+Hx, Hy+1:Ny+Hy+1, Hz+1:Nz+Hz]
                    global w[start:collect_c, :, :, nt] = w_temp[tstep][Hx+1:Nx+Hx, Hy+1:Ny+Hy, Hz+1:Nz+Hz+1]
                    global ν[start:collect_c, :, :, nt] = ν_temp[tstep][Hx+1:Nx+Hx, Hy+1:Ny+Hy, Hz+1:Nz+Hz]
                    global t[nt] = times[tstep]
                end 
                i += 1
            end
        end
    end
    return (u=u, v=v, w=w, ν=ν, times=t, grid= grid1)
end

function plot_error(case1, case2, title, filename; fixed_step=false)
    times1 = case1.times;
    times2 = case2.times;
    @assert times1 ≈ times2 "Time points in the two simulations do not match."

    # w is defined on the staggered grid, with points 1 and Nz being zero-value BCs
    # u and v on their staggered grids are periodic, and thus not zero.
    ν_err = (case2.ν .- case1.ν) ./ case1.ν;
    u_err = (case2.u .- case1.u) ./ case1.u;
    v_err = (case2.v .- case1.v) ./ case1.v;
    w_err = (case2.w .- case1.w) ./ case1.w;

    # ν_min = vec(minimum(abs, ν_err, dims=(1, 2, 3)));
    ν_avg = vec(mean(abs, ν_err, dims=(1, 2, 3)));
    ν_max = vec(maximum(abs, ν_err, dims=(1, 2, 3)));

    # u_min = vec(minimum(abs, u_err, dims=(1, 2, 3)));
    u_avg = vec(mean(abs, u_err, dims=(1, 2, 3)));
    u_max = vec(maximum(abs, u_err, dims=(1, 2, 3)));

    # v_min = vec(minimum(abs, v_err, dims=(1, 2, 3)));
    v_avg = vec(mean(abs, v_err, dims=(1, 2, 3)));
    v_max = vec(maximum(abs, v_err, dims=(1, 2, 3)));

    # w_min = vec(minimum(abs, w_err, dims=(1, 2, 3)));
    w_avg = vec(mean(abs, w_err, dims=(1, 2, 3)));
    w_max = vec(maximum(abs, w_err, dims=(1, 2, 3)));
    w_avg[isnan.(w_avg)] .= 0.0
    w_max[isnan.(w_max)] .= 0.0

    yscale = :log10
    
    y_max = maximum(maximum, [ν_max, u_max, v_max, w_max])
    y_min = minimum(minimum, [ν_avg[2:end], u_avg[2:end], v_avg[2:end], w_avg[2:end]])
    
    ytlo = floor(log10(y_min))
    ythi = ceil(log10(y_max))
    ylims = 10.0 .^(ytlo, ythi)
    @show ytlo, ythi, ylims

    if ytlo == -Inf
        @info "$title\n has zero maximum error in all fields!"
        yscale = :identity
        yticks = :auto 
        legend = :topright
    else
        yticks = 10.0 .^ (Int(ytlo) + 1 : 2 : Int(ythi) - 1)
    end

    if !fixed_step
        t = times1 / 3600.0
        xlabel = "time (hours)"
        legend = :best
    else
        t = 0:10
        xlabel = "Iterations"
        legend = :best
    end

    dpi = 300
    figsize = Int.((2.0, 1.75) .* dpi)

    plot(t, u_max; color=:blue, linestyle=:solid, linewidth=1.5, marker=:none, label="u max",
        title, xlabel, ylabel="Pointwise Abs Rel Error", legend, 
        yscale, ylims, yticks, minorgrid=false, size=figsize, dpi)
    
    plot!(t, u_avg; color=:blue, linestyle=:dash, linewidth=2, marker=:none, label="u avg")
    plot!(t, v_max; color=:purple, linestyle=:solid, linewidth=1.5, marker=:none, label="v max")
    plot!(t, v_avg; color=:purple, linestyle=:dash, linewidth=2, marker=:none, label="v avg")
    plot!(t, w_max; color=:green, linestyle=:solid, linewidth=1.5, marker=:none, label="w max")
    plot!(t, w_avg; color=:green, linestyle=:dash, linewidth=2, marker=:none, label="w avg")
    plot!(t, ν_max; color=:red, linestyle=:solid, linewidth=1.5, marker=:none, label="ν max")
    plot!(t, ν_avg; color=:red, linestyle=:dash, linewidth=2, marker=:none, label="ν avg")
    savefig(filename)
end
case1 = "outputs/forcing"
case2 = "outputs/sgs"
case1 = load_data(case1);
case2 = load_data(case2);
plot_error(case1, case2, "User Forcing Function vs Oceananigans Closure", "forcing_vs_sgs.png")