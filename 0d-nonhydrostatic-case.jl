using Pkg
#using MPI
#using CUDA
using Statistics
using Printf
using Random
Pkg.develop(path="/Users/annapauls/.julia/dev/Oceananigans.jl-main") #this will call for my version of Oceananigans locally 
Pkg.develop(path="/Users/annapauls/.julia/dev/OceanBioME.jl-main") #this will call for my version of OceanBioME locally
using Oceananigans, OceanBioME
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BuoyancyFormulations: g_Earth
using OceanBioME: Biogeochemistry, CarbonateChemistry
#using Oceananigans.DistributedComputations
#include("cc.jl")
#using .CC #: CarbonateChemistry #local module
#include("strang-rk3.jl") #local module
#using .SRK3
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
p = Params(1, 1, 1, 1.0, 1.0, 1.0, 5.3e-9, 33.0, 0.0, 4200.0, 1000.0, 0.01, 25.0, 2.0e-4, 5.75, 0.3)

grid = RectilinearGrid(; size=(p.Nx, p.Ny, p.Nz), extent=(p.Lx, p.Ly, p.Lz))
#grid = RectilinearGrid(arch; size=(p.Nx, p.Ny, p.Nz), extent=(p.Lx, p.Ly, p.Lz))

biogeochemistry = CarbonateChemistry(; grid, scale_negatives = true)
@show biogeochemistry
#DIC_bcs = FieldBoundaryConditions(top = GasExchange(; gas = :CO₂, temperature = (args...) -> p.T0, salinity = (args...) -> 35))

model = NonhydrostaticModel(; grid, buoyancy, #coriolis,
                            advection = UpwindBiased(order=1),
                            biogeochemistry, 
                            timestepper = :SplitCCRungeKutta3,
                            closure = nothing)#, CO₂ = DIC_bcs)) 
@show model

perturb = 1e3
set!(model, BOH₃ = 2.97e-4 * 1e6, BOH₄ = 1.19e-4 * 1e6, CO₂ = 7.57e-6 * 1e6 * perturb, CO₃ = 3.15e-4 * 1e6, HCO₃ = 1.67e-3 * 1e6, OH = 9.6e-6 * 1e6, T=25, S = 35)
simulation = Simulation(model, Δt=0.05, stop_time = 60seconds) #stop_time = 96hours,
@show simulation

output_interval = 0.05seconds

BOH₃ = model.tracers.BOH₃
BOH₄ = model.tracers.BOH₄
CO₂ = model.tracers.CO₂
CO₃ = model.tracers.CO₃
HCO₃ = model.tracers.HCO₃
OH = model.tracers.OH

simulation.output_writers[:fields] = JLD2Writer(model, (; BOH₃, BOH₄, CO₂, CO₃, HCO₃, OH),
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "split-testing.jld2", #$(rank)
                                                      overwrite_existing = true)

function progress(simulation)

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   prettytime(simulation.run_wall_time))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100000))
@info "Running the model..."
run!(simulation)