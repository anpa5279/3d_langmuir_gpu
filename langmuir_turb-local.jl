using Pkg
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
#using Oceananigans.BuoyancyFormulations: g_Earth #using Oceananigans: defaults #
using Oceananigans.DistributedComputations
using Oceananigans.TurbulenceClosures: Smagorinsky
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.AbstractOperations: ∂x, ∂y, ∂z
using Oceananigans.Operators: div_xyᶜᶜᶜ
const Nx = 16        # number of points in each of x direction
const Ny = 16        # number of points in each of y direction
const Nz = 16        # number of points in the vertical direction
const Lx = 320    # (m) domain horizontal extents
const Ly = 320    # (m) domain horizontal extents
const Lz = 96    # (m) domain depth 
const initial_mixed_layer_depth = 30.0 # m 
const Q = 5.0     # W m⁻², surface heat flux. cooling is positive
const cᴾ = 4200.0    # J kg⁻¹ K⁻¹, specific heat capacity of seawater
const ρₒ = 1026.0    # kg m⁻³, average density at the surface of the world ocean
const dTdz = 0.01  # K m⁻¹, temperature gradient
const T0 = 25.0    # C, temperature at the surface  
const S0 = 35.0    # ppt, salinity 
const β = 2.0e-4     # 1/K, thermal expansion coefficient
const u₁₀ = 5.75   # (m s⁻¹) wind speed at 10 meters above the ocean
const La_t = 0.3  # Langmuir turbulence number
grid = RectilinearGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)) #arch
# stokes drift
g_Earth = 9.81 #defaults.gravitational_acceleration
include("stokes.jl")
dusdz = Field{Nothing, Nothing, Center}(grid)
z_d = collect(-Lz + grid.z.Δᵃᵃᶜ/2 : grid.z.Δᵃᵃᶜ : -grid.z.Δᵃᵃᶜ/2)
dusdz_1d = dstokes_dz.(z_d, u₁₀)
set!(dusdz, reshape(dusdz_1d, 1, 1, :))
@show dusdz

# BCs
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q / (cᴾ * ρₒ)),
                                bottom = GradientBoundaryCondition(dTdz))
us = stokes_velocity(z_d, u₁₀)
u_f = La_t^2 * us[end]
const τx = -(u_f^2)# m² s⁻², surface kinematic momentum flux
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx), 
                                bottom = GradientBoundaryCondition(0.0))

v_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
# other forcing
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = β), constant_salinity = S0)

coriolis = FPlane(f=1e-4) # s⁻¹

model = NonhydrostaticModel(; grid, coriolis,
                            #advection = WENO(order=5),
                            timestepper = :RungeKutta3,
                            tracers = :T,
                            buoyancy = buoyancy,
                            closure = Smagorinsky(coefficient=0.1),#, Pr = 3.0),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=dusdz),
                            boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs)
                            )
@show model
# ICs
# --- PARAMETERS ---
ampv = 1e-3      # velocity amplitude [m/s]
ampt = 1e-3      # temperature amplitude
rng = Xoshiro(12345)

# Mixed-layer index (same as your code)
izi = Nz - Int(initial_mixed_layer_depth / Lz * Nz) + 1

# --- RANDOM STREAM FUNCTION ψ(x,y) ---
N_ml = Nz - izi + 1  # number of vertical levels in mixed layer
rand_maxtrix = randn(rng, Nx, Ny, N_ml)              # same as Fortran random_number()

# Extend ψ vertically but only in mixed layer
Ψ = zeros(Nx, Ny, Nz)
Ψ[:, :, izi:Nz] .= rand_maxtrix

# --- TAKE HORIZONTAL DERIVATIVES ---
# We use Oceananigans' built-in operators
psi_field = Field{Center, Center, Center}(grid)
set!(psi_field, Ψ)
∂ψ∂x = compute!(∂x(psi_field))   # returns a CCC field
∂ψ∂y = compute!(∂y(psi_field))

uprime = -∂ψ∂y.arg.data.parent[grid.Hx:Nx+grid.Hx-1, grid.Hy:Ny+grid.Hy-1, grid.Hz:Nz+grid.Hz-1]
vprime =  ∂ψ∂x.arg.data.parent[grid.Hx:Nx+grid.Hx-1, grid.Hy:Ny+grid.Hy-1, grid.Hz:Nz+grid.Hz-1]

# Normalize amplitude exactly like NCAR-LES
vmax = maximum(sqrt.(uprime.^2 .+ vprime.^2))
fac = ampv / vmax
uprime .*= fac
vprime .*= fac

# --- FULL INITIAL CONDITIONS ---
# Add mean profile us(z) just like your existing code
u_i = permutedims(us .* ones(Nz, Nx, Ny), [2, 3, 1]) .+ uprime
v_i = vprime

# Temperature IC
T_i = fill(T0, Nx, Ny, Nz)
T_i[:, :, izi:Nz] .+= ampt .* Ψ[:, :, izi:Nz]

# --- ASSIGN FIELDS ---
uᵢ = Field{Face, Center, Center}(grid)
vᵢ = Field{Center, Face, Center}(grid)
Tᵢ = Field{Center, Center, Center}(grid)

set!(uᵢ, u_i)
set!(vᵢ, v_i)
set!(Tᵢ, T_i)

fill_halo_regions!(uᵢ, u_bcs)
fill_halo_regions!(vᵢ, v_bcs)
fill_halo_regions!(Tᵢ, T_bcs)

#r_z(z) = z > - initial_mixed_layer_depth ? randn(Xoshiro()) : 0.0 
#T_i(x, y, z) = z > - initial_mixed_layer_depth ? (T0 + dTdz * model.grid.Lz * 1e-6 * r_z(z)) : T0 + dTdz * (z + initial_mixed_layer_depth) 
#ampv = 1.0e-3 # m s⁻¹ 
#ue(x, y, z) = r_z(z) * ampv 
#u_i(x, y, z) = -ue(x, y, z) + stokes_velocity(z, u₁₀)
#v_i(x, y, z) = ue(x, y, z)
#uᵢ = Field{Face, Center, Center}(grid)
#set!(uᵢ, u_i)
#fill_halo_regions!(uᵢ, u_bcs)
#vᵢ = Field{Center, Face, Center}(grid)
#set!(vᵢ, v_i)
#fill_halo_regions!(vᵢ, v_bcs)
#Tᵢ = Field{Center, Center, Center}(grid)
#set!(Tᵢ, T_i)
#fill_halo_regions!(Tᵢ, T_bcs)

set!(model, w=0.0, u=uᵢ, v=vᵢ, T=Tᵢ) 
@show "ICs set"
simulation = Simulation(model, Δt=30.0, stop_time=240*hours)
@show simulation

function progress(simulation)
    u, v, w = simulation.model.velocities

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(simulation.run_wall_time))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))

conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, max_Δt=30seconds)

#output files
function save_IC!(file, model)
    file["IC/friction_velocity"] = u_f
    file["IC/stokes_velocity"] = us
    file["IC/wind_speed"] = u₁₀
    return nothing
end

output_interval = 2.4*hours

u, v, w = model.velocities
T = model.tracers.T
W = Average(w, dims=(1, 2))
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
T_avg = Average(T, dims=(1, 2))

simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T),
                                                    schedule = TimeInterval(output_interval),
                                                    filename = "langmuir_turbulence_fields.jld2", #$(rank)
                                                    overwrite_existing = true,
                                                    with_halos = false,
                                                    array_type = Array{Float64},
                                                    init = save_IC!)
                                                      
T = Average(T, dims=(1, 2))
simulation.output_writers[:averages] = JLD2Writer(model, (; U, V, W, T),
                                                    schedule = AveragedTimeInterval(output_interval, window=output_interval),
                                                    filename = "langmuir_turbulence_averages.jld2",
                                                    overwrite_existing = true,
                                                    with_halos = false,
                                                    array_type = Array{Float64})

run!(simulation)#; pickup = true)