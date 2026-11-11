using Pkg
using JLD2
using Random
using Printf
#using SpecialFunctions # for erf()
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Units: minute, minutes, hours, seconds

Lx = Ly = 64           # (m) domain horizontal extents

Lz = 128             # (m) domain depth 
MLD = 40.0          # m, mixed layer depth
dTdz = 0.01       # K m⁻¹, temperature gradient
alpha = 2.0e-4      # 1/K, thermal expansion coefficient
rp = 4.0           # m, radius of surface buoyancy flux
area = (2*rp)^2 # square inlet condition
T0 = 25.0 + 273.15 # K, temperature at the surface
min_step = 0.01
wp = -0.001 # m/s, vertical velocity for surface buoyancy flux
Ln = rp #m
# constant functions
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
T_no_noise(x, y, z) = z > - MLD ? T0 : T0 + dTdz * (z + MLD)

Random.seed!(1234) 
r_z(z) = randn()*exp(z/rp)
T_noise(x, y, z) = z > - MLD ? T0 + dTdz * r_z(z) : T0 + dTdz * (z + MLD)

# buoyancy
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = alpha))

# calculating mass relative to morton length scale
g = buoyancy.gravitational_acceleration
beta = buoyancy.equation_of_state.haline_contraction
N = (dTdz * alpha * g)^(1/2)
F0 = (Ln * N^(3/4))^4
Sj = F0 / (g * beta * abs(wp) * pi * rp^2)

# BCs
@inline gauss(x, y) = exp(-((x^2 + y^2)/(2*rp^2)))
u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0), 
                                bottom = GradientBoundaryCondition(0.0))
T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),
                                bottom = GradientBoundaryCondition(dTdz))

#variations
dxs = [0.5,]
noise = [true, false]
noise_names = ["with noise in T IC", "without noise in T IC", ]
advection_schemes = [WENO(order = 5), WENO(order = 9)]
names = ["WENO5", "WENO9"]
for (noise_name, noise_opt) in zip(noise_names, noise)
    if noise_opt
        Tᵢ = T_noise
    else
        Tᵢ = T_no_noise
    end
    for (name, scheme) in zip(names, advection_schemes)
        for dx in dxs
            if noise_opt
                Random.seed!(1234) #resetting seed so noise is consistent
            end
            Nx = Ny = Int(Lx/dx)
            Nz = Int(Lz/dx)
            dir = "localoutputs/gauss testing/case2/$(noise_name)/$(name)/dx$(dx)/"
            @show dir

            println("Running simulation $(noise_name) with dx=$(dx) (Nx = $(Nx), Ny = $(Ny), Nz = $(Nz)) and scheme=$(scheme)")

            # defining grid
            grid = RectilinearGrid(; size=(Nx, Ny, Nz), x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = (-Lz, 0))

            # BCs
            gaus_area_cont = pi*rp^2 #*erf(Lx/(2*rp))*erf(Ly/(2*rp)) #erf produces 1
            gaus_area_discr = sum(exp.(-(repeat(grid.xᶜᵃᵃ[1:Nx], 1, Ny).^2 + transpose(repeat(grid.yᵃᶜᵃ[1:Ny], 1, Nx)).^2)./(rp^2)))*(Lx/Nx)*(Ly/Ny)
            factor = gaus_area_cont/gaus_area_discr #ensuring buoyancy flux is the same across cases regardless of resolution
            @inline w_surface(x, y, t)=wp*factor*gauss(x, y)

            w_bottom = wp * 2*pi*rp^2 / (Lx*Ly) 

            w_bcs = FieldBoundaryConditions(top = OpenBoundaryCondition(w_surface;  scheme = PerturbationAdvection(; inflow_timescale = 0.0, outflow_timescale = 0.0)), 
                                            bottom = OpenBoundaryCondition(w_bottom;  scheme = PerturbationAdvection(; inflow_timescale = 0.0, outflow_timescale = 0.0)))

            @inline S_surface(x, y, t)=Sj*factor*gauss(x, y)

            S_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(S_surface), 
                                            bottom = GradientBoundaryCondition(0.0))
            ## defining model
            model = NonhydrostaticModel(grid;
                                        buoyancy, 
                                        advection = scheme,
                                        tracers = (:T, :S,),
                                        timestepper = :RungeKutta3,
                                        boundary_conditions = (u = u_bcs, v = v_bcs, S=S_bcs, T=T_bcs, w=w_bcs),
                                        )
            @show model
            ## ICs
            set!(model, u=0.0, v=0.0, T=Tᵢ, S=0.0)

            # defining simulation
            simulation = Simulation(model, Δt=0.5/0.1*Lz/Nz, stop_time = 2hours, minimum_relative_step = 0.01) 
            ## progress function
            simulation.callbacks[:progress] = Callback(progress, IterationInterval(500))

            ## updating cfl every time step
            conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, diffusive_cfl = 1.0, min_change = 0.1, max_change = 1.5, min_Δt = min_step, max_Δt=30seconds) #ensrues cfl is updated ever iteration
            ## output files
            output_interval = 0.2hours
            u, v, w = model.velocities
            T = model.tracers.T
            S = model.tracers.S

            simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T, S),
                                                            array_type = Array{Float64},
                                                            schedule = TimeInterval(output_interval),
                                                            filename = "fields.jld2",
                                                            dir = dir,
                                                            overwrite_existing = true)

            simulation.output_writers[:centerline] = JLD2Writer(model, (; u, v, w, T, S), # test within if statement and outside of 
                                                            indices = (Int(grid.Nx/2):Int(grid.Nx/2+1), Int(Ny/2):Int(Ny/2+1), :),
                                                            array_type = Array{Float64},
                                                            schedule = TimeInterval(output_interval/100),
                                                            filename = "centerline.jld2",
                                                            dir = dir,
                                                            overwrite_existing = true)

            v_avg = Average(v, dims=(1, 2))
            w_avg = Average(w, dims=(1, 2))
            T_avg = Average(T, dims=(1, 2))
            S_avg = Average(S, dims=(1, 2))
            u_avg = Average(u, dims=(1, 2))
            simulation.output_writers[:xy_avg] = JLD2Writer(model, (; u_avg, v_avg, w_avg, T_avg, S_avg), # test within if statement and outside of
                                                            with_halos=false,
                                                            array_type = Array{Float64},
                                                            schedule = TimeInterval(output_interval/100),
                                                            filename = "xy_avg.jld2",
                                                            dir = dir,
                                                            overwrite_existing = true)

            run!(simulation)

            # Save grid metadata to a separate file (rank 0 only)
            jldopen("$(dir)/grid_info.jld2", "w") do file
                file["grid/x"]      = grid.xᶜᵃᵃ
                file["grid/y"]      = grid.yᵃᶜᵃ
                file["grid/z"]      = grid.z.cᵃᵃᶜ
                file["grid/Δx"]     = grid.Δxᶜᵃᵃ
                file["grid/Δy"]     = grid.Δyᵃᶜᵃ
                file["grid/Δz"]     = grid.z.Δᵃᵃᶜ
                file["grid/Nx"]     = Nx
                file["grid/Ny"]     = Ny
                file["grid/Nz"]     = Nz
                file["grid/Lx"]     = Lx
                file["grid/Ly"]     = Ly
                file["grid/Lz"]     = Lz
                file["grid/H"]      = model.grid.Hz
            end
        end
    end
end