using Pkg
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minute, minutes, hours, seconds
## simulation parameters
Nx = 64
Ny = 64
Nz = 64
Lx = 320            # (m) domain horizontal extents
Ly = 320            # (m) domain horizontal extents
Lz = 96             # (m) domain depth 
MLD = 30.0          # m, mixed layer depth
dTdz = 0.01         # K m⁻¹, temperature gradient
alpha = 2.0e-4      # 1/K, thermal expansion coefficient
rp = 10.0           # m, radius of surface buoyancy flux
T0 = 25.0           # C, temperature at the surface
min_step = 0.1
wp = -0.001 # m/s, vertical velocity for surface buoyancy flux
Sj = 0.1 # g/kg, tracer mass 

# defining grid
grid = RectilinearGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
@show grid

# buoyancy
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = alpha))
#beta = buoyancy.equation_of_state.haline_contraction

# BCs
u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),
                                bottom = GradientBoundaryCondition(dTdz))
S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(sflux), 
                                bottom = GradientBoundaryCondition(0.0))

# closure
Re = 3000
w_max = 0.10747783287769483
visc = w_max*Lz/Re # 1.0e-5 # m² s⁻¹
sgs = ScalarDiffusivity(ν=visc, κ=visc)

## defining model
model = NonhydrostaticModel(grid;
                            buoyancy, 
                            advection = WENO(),
                            tracers = (:T, :S,),
                            timestepper = :RungeKutta3,
                            boundary_conditions = (u = u_bcs, v = v_bcs, S=S_bcs, T=T_bcs),
                            closure = sgs
                            )
@show model
## ICs
izi = Nz - Int(MLD / Lz * Nz) + 1 # Mixed-layer index (same as your code)
rand_matrix = randn(Xoshiro(123), Nx, Ny, Nz) #.* reshape(exp.(grid.z.cᵃᵃᶜ[1:Nz] ./ 4), 1, 1, Nz)
rand_matrix = permutedims(rand_matrix, (2, 1, 3))
#uᵢ = wp .* rand_matrix
#vᵢ = -wp .* rand_matrix
r(x, y, z) = begin
    i = clamp(fld(Int(round(x / Lx * Nx)) + 1, 1), 1, Nx)
    j = clamp(fld(Int(round(y / Ly * Ny)) + 1, 1), 1, Ny)
    k = clamp(fld(Int(round((-z) / Lz * Nz)) + 1, 1), 1, Nz)

    rand_matrix[i, j, k] * exp(z/4)
end

#r(x, y, z) = exp(z/4)*(randn(Xoshiro())) 
uᵢ(x, y, z) = wp * r(x, y, z)
vᵢ(x, y, z) = -wp * r(x, y, z)
#uᵢ(x, y, z) = z > - MLD ? wp : 0.0
#vᵢ(x, y, z) = z > - MLD ? -wp : 0.0
#Tᵢ = ones(Nx, Ny, Nz) .* T0 +dTdz * Lz * 1e-6 * rand_matrix
#Tᵢ[:, :, 1:izi-1] = Tᵢ[:, :, 1:izi-1] .+ reshape(dTdz .* grid.z.cᵃᵃᶜ[1:izi-1], 1, 1, izi-1) 
Tᵢ(x, y, z) = z > - MLD ? T0 : 
                T0 + dTdz * (z + MLD) + dTdz * Lz * 1e-6 * r(x, y, z)

set!(model, u=uᵢ, v=vᵢ, T=Tᵢ, S=0.0)

# defining simulation
simulation = Simulation(model, Δt=min_step, stop_time = 12hours) 
@show simulation
## progress function
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
simulation.callbacks[:progress] = Callback(progress, IterationInterval(500))
## updating cfl every time step
conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, diffusive_cfl = 1.0, min_Δt = min_step, max_Δt=30seconds) #ensrues cfl is updated ever iteration
## output files
output_interval = 0.2hours
u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
P_static = model.pressures.pHY′
P_dynamic = model.pressures.pNHS
rel_path = "localoutputs/rand function within tranposed xy matrix/"
simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T, S, P_static, P_dynamic),
                                                    with_halos=false,
                                                    dir = rel_path, 
                                                    array_type = Array{Float64},
                                                    schedule = TimeInterval(output_interval),
                                                    filename = "fields.jld2",
                                                    overwrite_existing = true)#, init = save_grid!)# including = [default_included_properties(model), grid])

# running the simulation
run!(simulation)