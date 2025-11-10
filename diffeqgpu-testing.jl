using DiffEqGPU, OrdinaryDiffEq, StaticArrays, CUDA

include("cc.jl")
function cc_cell(du, u, p, t)
    T = p[1]
    CO2 = u[1]
    HCO3 = u[2]
    CO3 = u[3]
    H =  H_qss(a1, b1, a3, b3, a5, b5, CO₂, HCO₃, CO₃, OH)
    OH = u[4]
    BOH3 = u[5]
    BOH4 = u[6]

    S = 35.0
    #setting up parameters
    K1 = K_1(T, S)
    K2 = K_2(T, S)
    Kw = K_w(T, S)
    Kb = K_b(T, S)

    #Zeebe and Wolf Gladrow 2001
    A1 = 4.70e7 / 1e6 # kg/umol/s
    E1 = 23.2 # kJ/mol
    A7 = 4.58e10 / 1e6 # kg/umol/s
    E7 = 20.8 # kJ/mol
    A8 = 3.05e10 / 1e6 # kg/umol/s
    E8 = 20.8 # kJ/mol

    a1 = alpha1(T)
    b1 = beta1(a1, K1)
    a2 = alpha2(A1, E1, T)
    b2 = beta2(a2, Kw, K1)
    a3 = 5e10 / 1e6 # kg/umol/s, Dickson and Goyet 1994
    b3 = beta3(a3, K2)
    a4 = 6.0e9 / 1e6 # kg/umol/s, Dickson and Goyet 1994
    b4 = beta4(a4, Kw, K2)
    a5 = 1.40e-3 * 1e6 # umol/kg/s, Dickson and Goyet 1994
    b5 = beta5(a5, Kw)
    a6 = alpha6(A7, E7, T)
    b6 = beta6(a6, Kw, Kb)
    a7 = alpha7(A8, E8, T)
    b7 = beta7(a7, K2, Kb)
    vars = (a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7)

    du[1] = CarbonateChemistry_CO2(CO2, HCO3, CO3, OH, BOH3, BOH4, H, vars)
    du[2] = CarbonateChemistry_HCO3(CO2, HCO3, CO3, OH, BOH3, BOH4, H, vars)
    du[3] = CarbonateChemistry_CO3(CO2, HCO3, CO3, OH, BOH3, BOH4, H, vars)
    du[4] = CarbonateChemistry_OH(CO2, HCO3, CO3, OH, BOH3, BOH4, H, vars)
    du[5] = CarbonateChemistry_BOH3(CO2, HCO3, CO3, OH, BOH3, BOH4, H, vars)
    du[6] = -du[5] #BOH4

end
N = 8
perturb = 1e3
CO2 = fill(7.57e0 * perturb, N, N, N)
CO3 = fill(3.15e2, N, N, N)
HCO3 = fill(1.67e3, N, N, N)
OH = fill(9.6e0, N, N, N)
BOH3 = fill(2.97e2, N, N, N)
BOH4 = fill(1.19e2, N, N, N)
T = fill(25.0, N, N, N)

CO2_array = reshape(CO2, :)
HCO3_array = reshape(HCO3, :)
CO3_array = reshape(CO3, :)
OH_array = reshape(OH, :)
BOH3_array = reshape(BOH3, :)
BOH4_array = reshape(BOH4, :)
T_array = reshape(T, :)

u0_array = hcat(CO2_array, HCO3_array, CO3_array, OH_array, BOH3_array, BOH4_array)'  # 6 x N^3
p_array = hcat(T_array)'  # 1 x N^3
u0_array_d = CuArray(u0_array)  # 6 x N^3
p_array_d = CuArray(p_array)    # 1 x N^3
tspan = (0.0f0, 10.0f0)
@show tspan
prob = ODEProblem(cc_cell, u0_array, tspan, p_array)
@show "done with prob"
prob_func = function(prob, i, repeat)
    remake(prob, u0 = u0_array[:, i], p = p_array[:, i])  # CPU arrays
end
@show "done with prob func"

cc_prob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
@show "done with monte prob"

sol = solve(cc_prob, GPURodas4(), EnsembleGPUKernel(CUDA.CUDABackend()),
            trajectories = N^3, save_everystep=false)