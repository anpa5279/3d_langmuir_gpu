using OceanBioME, Oceananigans
using Oceananigans.Units
using Oceananigans.BuoyancyFormulations: g_Earth
using Printf
using MPI
using CUDA
include("cc.jl")
using .CC #: CarbonateChemistry #local module
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
p = Params(32, 32, 32, 320.0, 320.0, 96.0, 5.3e-9, 33.0, 0.0, 4200.0, 1000.0, 0.01, 25.0, 2.0e-4, 5.75, 0.3)

grid = BoxModelGrid()
clock = Clock(time = zero(grid))

biogeochemistry = CarbonateChemistry(; grid)

model = BoxModel(; biogeochemistry, clock)

set!(model, BOH₃ = 2.97e-4, BOH₄ = 1.19e-4, CO₂ = 20.0, CO₃ = 3.15e-4, HCO₃ = 1.67e-3, OH = 9.6e-6, T=25, S = 35)#BOH₃ = 2.97e-4, BOH₄ = 1.19e-4, CO₂ = 7.57e-6, CO₃ = 3.15e-4, HCO₃ = 1.67e-3, OH = 9.6e-6, T=25, S = 35)

simulation = Simulation(model, Δt=1e-8, stop_time = 60seconds) #stop_time = 96hours,
@show simulation

output_interval = 3seconds # 15 minutes

BOH₃ = model.fields.BOH₃
BOH₄ = model.fields.BOH₄
CO₂ = model.fields.CO₂
CO₃ = model.fields.CO₃
HCO₃ = model.fields.HCO₃
OH = model.fields.OH

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; BOH₃, BOH₄, CO₂, CO₃, HCO₃, OH),
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "outputs/box_model.jld2", #$(rank)
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

simulation.callbacks[:progress] = Callback(progress, IterationInterval(3e7))
@info "Running the model..."
run!(simulation)