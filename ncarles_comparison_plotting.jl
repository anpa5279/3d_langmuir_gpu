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
p = Params(128, 128, 160, 320.0, 320.0, 96.0, 5.3e-9, 33.0, 0.0, 4200.0, 1000.0, 0.01, 17.0, 2.0e-4, 5.75, 0.3)
grid = RectilinearGrid(size = (p.Nx, p.Ny, p.Nz), extent = (p.Lx, p.Ly, p.Lz))
function VKE(a, u_f)
    nz = length(a[1, 1, :, 1])
    ny = length(a[1, :, 1, 1])
    nx = length(a[:, 1, 1, 1])
    a_avg_xy = Statistics.mean(a, dims=(1, 2))
    a_prime = a .- a_avg_xy 
    a_prime2 = a_prime.^2
    aprime2_norm = Array{Float64}(undef, nz)
    aprime2_norm[:] = Statistics.mean(a_prime2, dims=(1, 2)) / (u_f^2)
    return aprime2_norm
end
function load_ncar_data(p, nt)
    i = 2 * nt - 1
    u_ncar = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz))
    v_ncar = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz))
    w_ncar = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz))
    T_ncar = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz))
    var_ncar = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz, nvar))
    @show size(var_ncar)

    if !occursin(".con", dtn[i]) && occursin("u.mp.", dtn[i])
        tmp = read("data/" * dtn[i])
        @show dtn[i]
        var_ncar = reshape(reinterpret(Float64, tmp), p.Nx, p.Ny, p.Nz, nvar)
        u_ncar = var_ncar[:,:,:,1]
        v_ncar = var_ncar[:,:,:,2]
        w_ncar = var_ncar[:,:,:,3]
        T_ncar = var_ncar[:,:,:,4]
        var_ncar = nothing
        tmp = nothing
        GC.gc()
        #getting u in yz plane
        u_yz = u_ncar[1, :, :]
        #getting averages of u, v, uw, vw, and T
        wu_ncar = Statistics.mean(w_ncar .* u_ncar, dims=(1, 2))
        wv_ncar = Statistics.mean(w_ncar .* v_ncar, dims=(1, 2))
        u_ncar_avg = Statistics.mean(u_ncar, dims=(1, 2))
        u_ncar = nothing
        GC.gc()
        v_ncar_avg = Statistics.mean(v_ncar, dims=(1, 2))
        v_ncar = nothing
        GC.gc()
        #calculating buoyancy
        B_ncar = g_Earth * p.β * Statistics.mean(T_ncar .- p.T0, dims=(1, 2))
        T_ncar = nothing
        GC.gc()
        
    end

    return u_yz, w_ncar, B_ncar, u_ncar_avg, v_ncar_avg, wu_ncar, wv_ncar
end
dtn = cd(readdir, "data")
nvar = 5
Nt = Int(length(dtn) / 2)
t_end = 0.28363754e06
times = range(0, stop=t_end, length=Nt)
z = range(0, stop=p.Lz, length=p.Nz)
#intializing arrays
wprime2_obs = zeros(p.Nz)# VKE vertical profile
u_ncarles = zeros(p.Ny, p.Nz)
w_ncarles = zeros(p.Nx, p.Ny, p.Nz)
B_ncarles = zeros(p.Nz)
U_ncarles = zeros(p.Nz)
V_ncarles = zeros(p.Nz)
wu_ncarles = zeros(p.Nz)
wv_ncarles = zeros(p.Nz)
time = Observable(0.0)

# plotting results
axis_kwargs = (xlabel="y (m)",
            ylabel="z (m)",
            aspect = AxisAspect(p.Lx/p.Lz),
            limits = ((0, p.Lx), (-p.Lz, 0)))
fig = Figure(size = (850, 850))

# w surface plane slice
wxy_title = @lift string("w(x, y, t), at z=-8 m and t = ", $time)
ax_wxy = Axis(fig[1, 1:2];
            xlabel = "x (m)",
            ylabel = "y (m)",
            aspect = DataAspect(),
            limits = ((0, grid.Lx), (0, grid.Ly)),
            title = wxy_title)
k = searchsortedfirst(znodes(grid, Face(); with_halos=true), -8)
#wxyₙ = w_ncarles[:, :, k]
wlims = (-0.02, 0.02)
hm_wxy = heatmap!(ax_wxy, w_ncarles[:, :, k];
                colorrange = wlims,
                colormap = :balance)
Colorbar(fig[1, 3], hm_wxy; label = "m s⁻¹")

# w yz plane slice
wxz_title = @lift string("w(x, z, t), at x=0 m and t = ", $time)
ax_wyz = Axis(fig[2, 1:2]; title = wxz_title, axis_kwargs...)
hm_wxz = heatmap!(ax_wyz, w_ncarles[1, :, :];
                colorrange = wlims,
                colormap = :balance)

Colorbar(fig[2, 3], hm_wxz; label = "m s⁻¹")

# u yz plane slice
uxz_title = @lift string("u(x, z, t), at x=0 m and t = ", $time)
ax_uyz = Axis(fig[3, 1:2]; title = uxz_title, axis_kwargs...)
ulims = (-0.1, 0.1)
hm_uyz = heatmap!(ax_uyz, u_ncarles;
                colorrange = ulims,
                colormap = :balance)

Colorbar(fig[3, 3], hm_uyz; label = "m s⁻¹")

# buoyancy with depth
ax_B = Axis(fig[1, 4:5];
            xlabel = "Buoyancy (m s⁻²)",
            ylabel = "z (m)",
            limits = ((-0.002, 0.0), nothing))
lines!(ax_B, B_ncarles)

# mean horizontal velocities with depth
ax_U = Axis(fig[2, 4:5];
            xlabel = "Velocities (m s⁻¹)",
            ylabel = "z (m)",
            limits = ((-0.2, 0.2), nothing))
lines!(ax_U, U_ncarles; label = L"\bar{u}")
lines!(ax_U, V_ncarles; label = L"\bar{v}")
axislegend(ax_U; position = :rb)

# momentum fluxes with depth
ax_fluxes = Axis(fig[3, 4:5];
                xlabel = "Momentum fluxes (m² s⁻²)",
                ylabel = "z (m)",
                limits = ((-4.0e-5, 4.0e-5), nothing))
wuₙ = view(wu_ncarles, :)
wvₙ = view(wv_ncarles, :)
lines!(ax_fluxes, wu_ncarles; label = L"\overline{wu}")
lines!(ax_fluxes, wv_ncarles; label = L"\overline{wv}")
axislegend(ax_fluxes; position = :rb)

#VKE
ax_fluxes = Axis(fig[4, 4:5];
                xlabel = L"\overline{w'²} / u★²",
                ylabel = "z (m)",
                limits = ((0.0, 5.0), nothing))
lines!(ax_fluxes, wprime2_obs, z; label = L"\overline{w'²} / u★²")
axislegend(ax_fluxes; position = :rb)

fig

frames = 1:length(times)

record(fig, "plotting_ncarles.mp4", frames, framerate=8) do i
    @show i
    time[] = times[i]
    u_yz, w_ncar, B_ncar, u_ncar_avg, v_ncar_avg, wu_ncar, wv_ncar = load_ncar_data(p, i)
    wprime2_obs = VKE(w_ncar, 0.5301e-02)
    u_ncarles = u_yz
    u_yz = nothing
    GC.gc()
    w_ncarles = w_ncar
    w_ncar = nothing
    GC.gc()
    B_ncarles = B_ncar
    B_ncar = nothing
    GC.gc()
    U_ncarles = u_ncar_avg
    @show maximum(U_ncarles), minimum(U_ncarles)
    u_ncar_avg = nothing
    GC.gc()
    V_ncarles = v_ncar_avg
    v_ncar_avg = nothing
    GC.gc()
    wu_ncarles = wu_ncar
    wu_ncar = nothing
    GC.gc()
    wv_ncarles = wv_ncar
    wv_ncar = nothing
    GC.gc()
end 

