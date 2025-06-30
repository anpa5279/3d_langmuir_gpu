module SRK3

using Oceananigans.Architectures: architecture
using Oceananigans: fields
using Oceananigans: AbstractModel, initialize!, prognostic_fields
using Oceananigans.TimeSteppers: AbstractTimeStepper
using KernelAbstractions: @kernel, @index
using Oceananigans.AbstractOperations: KernelFunctionOperation
import Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Utils: launch!
include("cc.jl")
import .CC: CarbonateChemistry #local module

export StrangRungeKutta3TimeStepper

TimeStepper(::Val{:StrangRungeKutta3}, args...; kwargs...) =
    StrangRungeKutta3TimeStepper(args...; kwargs...)

"""
    StrangRungeKutta3TimeStepper{FT, TG} <: AbstractTimeStepper

Hold parameters and tendency fields for a low storage, third-order Runge-Kutta-Wray
time-stepping scheme described by [LeMoin1991](@citet) wtih DiffEq package for chemistry.
"""
struct StrangRungeKutta3TimeStepper{FT, TG, TI} <: AbstractTimeStepper
                 γ¹ :: FT
                 γ² :: FT
                 γ³ :: FT
                 ζ² :: FT
                 ζ³ :: FT
                 Gⁿ :: TG
                 G⁻ :: TG
    implicit_solver :: TI
end

function StrangRungeKutta3TimeStepper(grid, prognostic_fields;
                                implicit_solver::TI = nothing,
                                Gⁿ::TG = map(similar, prognostic_fields),
                                G⁻     = map(similar, prognostic_fields)) where {TI, TG}

    γ¹ = 8 // 15
    γ² = 5 // 12
    γ³ = 3 // 4

    ζ² = -17 // 60
    ζ³ = -5 // 12

    FT = eltype(grid)

    return StrangRungeKutta3TimeStepper{FT, TG, TI}(γ¹, γ², γ³, ζ², ζ³, Gⁿ, G⁻, implicit_solver)
end

#####
##### Time steppping
#####
TimeStepper(::Val{:StrangRungeKutta3}, args...; kwargs...) =
    StrangRungeKutta3TimeStepper(args...; kwargs...)

function time_step!(model::AbstractModel{<:StrangRungeKutta3TimeStepper}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Be paranoid and update state at iteration 0, in case run! is not used:
    model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies = true)

    γ¹ = model.timestepper.γ¹
    γ² = model.timestepper.γ²
    γ³ = model.timestepper.γ³

    ζ² = model.timestepper.ζ²
    ζ³ = model.timestepper.ζ³

    first_stage_Δt  = γ¹ * Δt
    second_stage_Δt = (γ² + ζ²) * Δt
    third_stage_Δt  = (γ³ + ζ³) * Δt

    # Compute the next time step a priori to reduce floating point error accumulation
    tⁿ⁺¹ = next_time(model.clock, Δt)

    #input carboante chemistry calculation here
    model.clock.stage = 0
    split_cc!(model, Δt, γⁿ, ζⁿ)
    #
    # First stage
    #

    strang_rk3_substep!(model, Δt, γ¹, nothing)

    tick!(model.clock, first_stage_Δt; stage=true)
    model.clock.last_stage_Δt = first_stage_Δt

    calculate_pressure_correction!(model, first_stage_Δt)
    pressure_correct_velocities!(model, first_stage_Δt)

    cache_previous_tendencies!(model)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, first_stage_Δt)

    #
    # Second stage
    #

    strang_rk3_substep!(model, Δt, γ², ζ²)

    tick!(model.clock, second_stage_Δt; stage=true)
    model.clock.last_stage_Δt = second_stage_Δt

    calculate_pressure_correction!(model, second_stage_Δt)
    pressure_correct_velocities!(model, second_stage_Δt)

    cache_previous_tendencies!(model)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, second_stage_Δt)

    #
    # Third stage
    #

    strang_rk3_substep!(model, Δt, γ³, ζ³)

    # This adjustment of the final time-step reduces the accumulation of
    # round-off error when Δt is added to model.clock.time. Note that we still use
    # third_stage_Δt for the substep, pressure correction, and Lagrangian particles step.
    corrected_third_stage_Δt = tⁿ⁺¹ - model.clock.time

    tick!(model.clock, third_stage_Δt)
    model.clock.last_stage_Δt = corrected_third_stage_Δt
    model.clock.last_Δt = Δt

    calculate_pressure_correction!(model, third_stage_Δt)
    pressure_correct_velocities!(model, third_stage_Δt)

    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, third_stage_Δt)

    #input carboante chemistry calculation here
    model.clock.stage = 0
    split_cc!(model, Δt, γⁿ, ζⁿ)

    return nothing
end

#####
##### Carbonate chemistry split
#####
function boxmodel_ode!(du, u, p, t)
    CO₂ = u[1]
    HCO₃ = u[2]
    CO₃ = u[3]
    OH = u[4]
    BOH₃ = u[5]
    BOH₄ = u[6]
    T = u[7]
    S = u[8]
    du[1] = bgc(Val(:CO₂),  x, y, z, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S)
    du[2] = bgc(Val(:HCO₃), x, y, z, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S)
    du[3] = bgc(Val(:CO₃),  x, y, z, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S)
    du[4] = bgc(Val(:OH),   x, y, z, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S)
    du[5] = bgc(Val(:BOH₃), x, y, z, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S)
    du[6] = bgc(Val(:BOH₄), x, y, z, t, CO₂, HCO₃, CO₃,  OH, BOH₃, BOH₄, T , S)
end
@kernel function update_cc!(grid, tracers)
    i, j, k = @index(Global, NTuple)
    c_0 = (tracers.CO₂[i, j, k],
             tracers.HCO₃[i, j, k],
             tracers.CO₃[i, j, k],
             tracers.OH[i, j, k],
             tracers.BOH₃[i, j, k],
             tracers.BOH₄[i, j, k],
             tracers.T[i, j, k],
             tracers.S[i, j, k])
    tspan = (0.0, Δt)
    prob = ODEProblem(boxmodel_ode!, c_0, tspan)
    sol = solve(prob, alg_hints = [:stiff], reltol = 1e-6, abstol = 1e-10, saveat = Δt) 
    
    @inbounds tracers.CO₂[i, j, k] = sol.u[1]
    @inbounds tracers.HCO₃[i, j, k] = sol.u[2]
    @inbounds tracers.CO₃[i, j, k] = sol.u[3]
    @inbounds tracers.OH[i, j, k] = sol.u[4]
    @inbounds tracers.BOH₃[i, j, k] = sol.u[5]
    @inbounds tracers.BOH₄[i, j, k] = sol.u[6]
end

function split_cc!!(model, Δt, γⁿ, ζⁿ)

    grid = model.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, update_cc!, grid, model.tracers)
    chem_tracers = NamedTuple{keys(model.tracers)[1:6]}(values(model.tracers)[1:end-2]) #excluding temperature and salinity
    for (i, field) in enumerate(chem_tracers)
        fill_halo_regions!(field)
    end
    return nothing
end

#####
##### Time stepping in each substep
#####

stage_Δt(Δt, γⁿ, ζⁿ) = Δt * (γⁿ + ζⁿ)
stage_Δt(Δt, γⁿ, ::Nothing) = Δt * γⁿ

function strang_rk3_substep!(model, Δt, γⁿ, ζⁿ)

    grid = model.grid
    arch = architecture(grid)
    model_fields = prognostic_fields(model)

    for (i, field) in enumerate(model_fields)
        kernel_args = (field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[i], model.timestepper.G⁻[i])
        launch!(arch, grid, :xyz, strang_rk3_substep_field!, kernel_args...; exclude_periphery=true)

        # TODO: function tracer_index(model, field_index) = field_index - 3, etc...
        tracer_index = Val(i - 3) # assumption

        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.diffusivity_fields,
                       tracer_index,
                       model.clock,
                       stage_Δt(Δt, γⁿ, ζⁿ)) #this is dependent on the turbulent closure, usually does nothing to the tracers. 
    end

    return nothing
end

@kernel function strang_rk3_substep_field!(U, Δt, γⁿ::FT, ζⁿ, Gⁿ, G⁻) where FT
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U[i, j, k] += convert(FT, Δt) * (γⁿ * Gⁿ[i, j, k] + ζⁿ * G⁻[i, j, k])
    end
end

@kernel function strang_rk3_substep_field!(U, Δt, γ¹::FT, ::Nothing, G¹, G⁰) where FT
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U[i, j, k] += convert(FT, Δt) * γ¹ * G¹[i, j, k]
    end
end

#how do I override using Oceananigans.Biogeochemistry: biogeochemical_transition in nonhydrostatic_tendency_kernel_functions.jl so it calls this function?
@inline function biogeochemical_transition(i, j, k, grid, bgc::CarbonateChemistry, 
                                           val_tracer_name, clock, fields)
    return zero(grid) 
end

end #module
