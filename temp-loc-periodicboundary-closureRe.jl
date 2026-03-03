using Pkg
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶠ
## simulation parameters
Lx = 320    # (m) domain horizontal extents
Ly = 320    # (m) domain horizontal extents
Lz = 96    # (m) domain depth 
initial_mixed_layer_depth = 30.0 # m 
w_max = 0.10747783287769483
ρₒ = 1026.0    # kg m⁻³, average density at the surface of the world ocean
dTdz = 0.01  # K m⁻¹, temperature gradient
T0 = 25.0    # C, temperature at the surface  
S₀ = 35.0    # ppt, salinity 
alpha = 2.0e-4     # 1/K, thermal expansion coefficient
Q = 5.0     # W m⁻², surface heat flux. cooling is positive
cᴾ = 4200.0    # J kg⁻¹ K⁻¹, specific heat capacity of seawater
ρₒ = 1026.0    # kg m⁻³, average density at the surface of the world ocean

# BCs
u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
g = Oceananigans.defaults.gravitational_acceleration

u_f = 0.001
b0 = -4e-1 # m s⁻²
T0_flux = (b0/(g*alpha)+T0) # K, surface temperature
Jᵇ = -u_f*T0_flux # m² s⁻³, surface temperature flux
@inline function Tflux_t(x, y, t) 
    if (t <= 6hours)
        σ = 10.0 # m
        return Jᵇ/(2*pi* σ^2) * exp(-(x-Lx/2)^2 / (2 * σ^2)) * exp(-(y-Ly/2)^2 / (2 * σ^2)) 
    else
        return 0.0
    end
end
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Tflux_t), 
                                    bottom = GradientBoundaryCondition(dTdz))

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = alpha), constant_salinity = S₀)
## ICs
r(x, y, z) = (randn(Xoshiro())) * exp(z/4)
uᵢ(x, y, z) = u_f * r(x, y, z)
vᵢ(x, y, z) = -u_f * r(x, y, z)

σ = 10.0 # m
plume(x, y, z) = T0_flux/sqrt((2*pi)^3* (σ^2)^3) * exp(-z^2 / (2 * σ^2)) * exp(-(x-Lx/2)^2 / (2 * σ^2)) * exp(-(y-Ly/2)^2 / (2 * σ^2)) 
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 + dTdz * Lz * 1e-6 * r(x, y, z) : #random noise in the mixed layer
                T0+ dTdz * (z + initial_mixed_layer_depth) + dTdz * Lz * 1e-6 * r(x, y, z)

    # closure
Re = 3000
path = "localoutputs/temperature for NBP/with closure Re $Re"
visc = w_max*Lz/Re # m² s⁻¹
sgs = ScalarDiffusivity(ν=visc, κ=visc)
@show sgs
for N in (128, )
    Nx = N
    Ny = N
    for Nz in (256,) #16, 32, 64, 128
        println("Running simulation with Nx = $Nx, Ny = $Ny, Nz = $Nz")
        ## referring to files with desiraed functions
        grid = RectilinearGrid(; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
        @show grid
        ## defining model
        model = NonhydrostaticModel(grid;
                                    buoyancy, 
                                    advection = WENO(),
                                    tracers = (:T,),
                                    timestepper = :RungeKutta3,
                                    boundary_conditions = (u = u_bcs, v = v_bcs, T=T_bcs),
                                    closure = sgs
                                    )
        @show model
        set!(model, u=uᵢ, v=vᵢ, T=Tᵢ)

        # defining simulation
        simulation = Simulation(model, Δt=30, stop_time = 12hours) 
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
        simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
        ## updating cfl every time step
        conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, min_Δt = 1.0, max_Δt=30seconds) #ensrues cfl is updated ever iteration
        ## output files
        output_interval = 0.2hours

        rel_path = "$path/flux temperature Nx = $Nx, Ny = $Ny, Nz = $Nz/"
        u, v, w = model.velocities
        T = model.tracers.T
        P_static = model.pressures.pHY′
        P_dynamic = model.pressures.pNHS
        simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T, P_static, P_dynamic),
                                                            dir = rel_path,  with_halos=false,
                                                            array_type = Array{Float64},
                                                            schedule = TimeInterval(output_interval),
                                                            filename = "fields.jld2", #$(rank)
                                                            overwrite_existing = true)

        # running the simulation
        run!(simulation)
    end
end