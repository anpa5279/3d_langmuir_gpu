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


Nranks = 4
fld_file="outputs/langmuir_turbulence_fields_0.jld2"
f = jldopen(fld_file)

#time and IC data 
u★ = f["IC"]["friction_velocity"]
#u_stokes = f["IC"]["stokes_velocity"]
#u₁₀ = f["IC"]["wind_speed"]
t_save = collect(keys(f["timeseries"]["t"]))
close(f)

Nt = length(t_save) - 1
@show Nt

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
B_data =  Array{Float64}(undef, (1, 1, p.Nz, Int(Nt/2)))
U_data = Array{Float64}(undef, (1, 1, p.Nz, Int(Nt/2)))
V_data = Array{Float64}(undef, (1, 1, p.Nz, Int(Nt/2)))
wu_data = Array{Float64}(undef, (1, 1, p.Nz + 1, Int(Nt/2)))
wv_data =   Array{Float64}(undef, (1, 1, p.Nz + 1, Int(Nt/2)))
B_data .= 0
U_data .= 0
V_data .= 0
wu_data .= 0
wv_data .= 0

for i in 0:Nranks-1
    println("Loading rank $i")

    fld_file="outputs/langmuir_turbulence_fields_$(i).jld2"
    averages_file="outputs/langmuir_turbulence_averages_$(i).jld2"

    f = jldopen(fld_file)
    B_temp = FieldTimeSeries(averages_file, "B")
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
    B_data .= B_data .+ B_temp.data[:, :, 1:p.Nz, 1:Int(Nt/2)]
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
    B_temp = nothing
    U_temp = nothing
    V_temp = nothing
    wu_temp = nothing
    wv_temp = nothing
    GC.gc() 
    close(f)
end

