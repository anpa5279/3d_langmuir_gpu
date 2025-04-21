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
grid = RectilinearGrid(size = (p.Nx, p.Ny, p.Nz), extent = (p.Lx, p.Ly, p.Lz), halo = (3, 3, 3))
Nranks = 4

#requirement parameters for initializing the matrices
fld_file="outputs/langmuir_turbulence_fields_0.jld2"
f = jldopen(fld_file)
t_temp = keys(f["timeseries"]["t"])
Nt = length(t_temp)
nn = 1
Nr = Int(p.Nx / Nranks)

times = Array{Float64}(undef, Nt)
w_data = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz + 1, Nt)) #because face value
u_data = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz, Nt))
b_data = Array{Float64}(undef, (p.Nx, p.Ny, p.Nz, Nt))

xrange = grid.Hx + 1 : grid.Hx + Nr
yrange = grid.Hy + 1 : grid.Hy + p.Ny
wrange = grid.Hz + 1 : grid.Hz + p.Nz + 1
urange = grid.Hz + 1 : grid.Hz + p.Nz
t_save = collect(keys(f["timeseries"]["t"]))
sort!(t_save)
w_all = [f["timeseries"]["w"][t][xrange, yrange, wrange] for t in t_save]
u_all = [f["timeseries"]["u"][t][xrange, yrange, urange] for t in t_save]
b_all = [f["timeseries"]["b"][t][xrange, yrange, urange] for t in t_save]
for k in 1:Nt
    @show k
    times[k] = f["timeseries"]["t"][t_save[k]]
    local w = w_all[k]
    local u = u_all[k]
    local b = b_all[k]

    w_data[nn:nn + Nr - 1, :, :, k] = w
    u_data[nn:nn + Nr - 1, :, :, k] = u
    b_data[nn:nn + Nr - 1, :, :, k] = b
end

close(f)
close(fld_file)
