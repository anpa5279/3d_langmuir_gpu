using ModelingToolkit, OrdinaryDiffEq, LinearAlgebra

function build_carbonate_system(; T=298.15, S=35.0)

    @parameters t a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 a6 b6 a7 b7 T S
    @variables CO2(t) HCO3(t) CO3(t) H(t) OH(t) BOH3(t) BOH4(t)

    eqs = [
        Differential(t)(CO2)  ~ - (a1 + a2 * OH) * CO2 + (b1 * H + b2 + b3 + a4 * OH + b7 * BOH4) * HCO3,
        Differential(t)(HCO3) ~ (a1 + a2 * OH) * CO2 - (b1 * H + b2 + b3 + a4 * OH + b7 * BOH4) * HCO3 + (a3 * H + b4 + a7 * BOH3) * CO3,
        Differential(t)(CO3)  ~ b3 * HCO3 - (a3 * H + b4) * CO3,
        Differential(t)(H)    ~ b1 * HCO3 - a3 * H * CO3 + b5 * OH - a5 * H + b6 * BOH4 - a6 * H * BOH3,
        Differential(t)(OH)   ~ a4 * HCO3 - b5 * OH + a5 * H,
        Differential(t)(BOH3) ~ b7 * HCO3 - a7 * BOH3 * CO3 + b6 * BOH4 - a6 * H * BOH3,
        Differential(t)(BOH4) ~ a6 * H * BOH3 - b6 * BOH4
    ]

    @named sys = ODESystem(eqs, t)

    return sys
end

function compute_jacobian_at_initial_condition()
    sys = build_carbonate_system()

    # Simplify and generate ODE function
    sys_simplified = structural_simplify(sys)
    f = ODEFunction(sys_simplified; jac=true)

    # Define parameters in order of sys_simplified.ps
    p_names = parameters(sys_simplified)
    p = ones(length(p_names))   # You can plug in real parameter values here

    # Define initial condition in order of sys_simplified.states
    u_names = states(sys_simplified)
    u0 = ones(length(u_names))  # Replace with realistic initial values

    # Evaluate Jacobian at t=0
    J_func = f.jac
    J = J_func(0.0, u0, p)

    println("Jacobian evaluated at t = 0:")
    display(J)
end

compute_jacobian_at_initial_condition()