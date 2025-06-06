
num_method = "rk3" # rk3, euler
#definiing the carbonate chemsitry reactions 
    T=25
    S = 35
    # carboante chemistry
    const R = 8.31446261815324 # kg⋅m²⋅s⁻²⋅K⁻1⋅mol⁻1
    #Zeebe and Wolf Gladrow 2001
    const A1 = 4.70e7 # kg/mol/s
    const E1 = 1000 * 23.2 # J/mol
    const A7 = 4.58e10 # kg/mol/s
    const E7 = 1000 * 20.8 # J/mol
    const A8 = 3.05e10 # kg/mol/s
    const E8 = 1000 * 20.8 # J/mol
    #Dickson and Goyet 1994

    #Dickson and Goyet 1994, who references Roy et al. 1993,  Dickson 1990, and Millero 1994
    const K1 = exp(-2307.1266 / (T + 273.15) + 2.83655 - 1.5529413 * log(T + 273.15) +
                (-4.0484 / (T + 273.15) - 0.20760841) * sqrt(S) + 0.08468345 * S -
                0.00654208 * S^1.5 + log(1 - 0.001005 * S))

    const K2 = exp(-3351.6106 / (T + 273.15) - 9.226508 - 0.2005743 * log((T + 273.15)) +
                (-23.9722 / (T + 273.15) - 0.106901773)* sqrt(S) + 0.1130822 * S -
                0.00846934 * S^1.5 + log(1 - 0.001005 * S))

    const Kw = exp(-13847.26 / (T + 273.15) + 148.9652 - 23.6521 * log((T + 273.15)) + 
                (118.67 / (T + 273.15) - 5.977 + 1.0495 * log((T + 273.15))) * sqrt(S) - 0.01615 * S)

    const Kb = exp((-8966.90 - 2890.53 * sqrt(S) - 77.942 * S + 1.728 * S^1.5 - 
                0.0996 * S^2) / (T + 273.15) + (148.0248 + 137.1942 * sqrt(S) + 
                1.62142 * S) + (-24.4344 - 25.085 * sqrt(S) -0.2474 * S) * log((T + 273.15)) 
                + 0.053105 * sqrt(S) * (T + 273.15))

    #Zeebe and Wolf Gladrow 2001
    const a1 = exp(1246.98 - 6.19e4 / (T + 273.15) - 183.0 * log(T + 273.15)) # kg/mol/s
    @show a1
    const b1 = a1/ K1 # 1/s
    @show b1
    const a2 = A1 * exp(-E1 / (R * (T + 273.15))) # kg/mol/s
    @show a2
    const b2 = a2 * Kw/ K1# 1/s
    @show b2
    const a3 = 5e10 # kg/mol/s
    @show a3
    const b3 = a3 * K2 # 1/s
    @show b3
    const a4 = 6.0e9 # kg/mol/s
    @show a4
    const b4 = a4 * Kw/ K2 # 1/s
    @show b4
    const a5 = 1.40e-3 # kg/mol/s
    @show a5
    const b5 = a5 / Kw # kg/mol/s
    @show b5
    const a6 = A7 * exp(-E8 / (R * (T + 273.15))) # kg/mol/s
    @show a6
    const b6 =  a6* Kw/ Kb # 1/s
    @show b6
    const a7 = A8 * exp(-E8 / (R * (T + 273.15))) # kg/mol/s
    @show a7
    const b7 = a7* K2/ Kb # kg/mol/s
    @show b7

    #updating tracers 
    @inline function dCO₂dt(CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
        #println("a1 = ", a1, " b1 = ", b1, " a2 = ", a2, " b2 = ", b2)
        if isnan(CO₂) error("CO₂ concentration is NaN") end
        dcdt = - (a1 + a2 * OH) * CO₂ + (b1 * H + b2) * HCO₃
        return dcdt
    end

    @inline function dHCO₃dt(CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
        #println("a3 = ", a3, " b3 = ", b3, " a4 = ", a4, " b4 = ", b4)
        if isnan(HCO₃) error("HCO₃ concentration is NaN") end
        dcdt = (a1 + a2 * OH) * CO₂ - (b1 * H + b2 + b3 + a4 * OH + b7 * BOH₄) * HCO₃ + (a3 * H + b4 + a7 * BOH₃) * CO₃
        return dcdt
    end

    @inline function dCO₃dt(CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
        if isnan(CO₃) error("CO₃ concentration is NaN") end
        dcdt = (b3 + a4 * OH + b7 * BOH₄) * HCO₃ - (a3 * H + b4 + a7 * BOH₃) * CO₃
        return dcdt
    end

    @inline function dHdt(CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
        #println("a5 = ", a5, " b5 = ", b5)
        if isnan(H) error("H concentration is NaN") end
        dcdt = a1 * CO₂ - (b1 * H - b3) * HCO₃ - a3 * H * CO₃ + (a5 - b5 * H * OH)
        return dcdt
    end

    @inline function dOHdt(CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
        #println("a6 = ", a6, " b6 = ", b6)
        if isnan(OH) error("OH concentration is NaN") end
        dcdt = - a2 * OH * CO₂ + (b2 - a4 * OH) * HCO₃ + b4 * CO₃ + (a5 - b5 * H * OH) - (a6 * OH * BOH₃ - b6 * BOH₄)
        return dcdt
    end

    @inline function dBOH₃dt(CO₂, HCO₃, CO₃, H, OH, BOH₃, BOH₄, T, S)
        #println("a7 = ", a7, " b7 = ", b7)
        if isnan(BOH₃) error("BOH₃ concentration is NaN") end
        if isnan(BOH₄) error("BOH₄ concentration is NaN") end
        dcdt = b7 * BOH₄ * HCO₃ - a7 * BOH₃ * CO₃ - (a6 * OH * BOH₃ - b6 * BOH₄)
        return dcdt
    end


tf = 1*60*60 # 1 hour
dt = 30 # seconds, output interval for the simulation
time = 0:dt:tf
nt = size(time)[1]

BOH₃ = Array{Float64}(undef, nt)
BOH₃[1] = 2.97e-4
BOH₄ = Array{Float64}(undef, nt)
BOH₄[1] = 1.19e-4
CO₂ = Array{Float64}(undef, nt)
CO₂[1] = 7.57e-6
CO₃ = Array{Float64}(undef, nt)
CO₃[1] = 3.15e-4
H = Array{Float64}(undef, nt)
H[1] = 6.31e-9
HCO₃ = Array{Float64}(undef, nt)
HCO₃[1] = 1.67e-3
OH = Array{Float64}(undef, nt)
OH[1] = 9.6e-6

#rk3 parameters
f = [8/15, 5/12, 3/4] #weights for current time step
b = [0, -17/60, -5/12] #weights for previous time step
if num_method == "rk3"
    for i in 1:nt-1
        println("\nN: ", i, ": Time: ", time[i], " seconds")
        println("CO₂: ", CO₂[i], " mol/m³")
        println("HCO₃: ", HCO₃[i], " mol/m³")
        println("CO₃: ", CO₃[i], " mol/m³")
        println("H: ", H[i], " mol/m³")
        println("OH: ", OH[i], " mol/m³")
        println("BOH₃: ", BOH₃[i], " mol/m³")
        println("BOH₄: ", BOH₄[i], " mol/m³")
        println("")

        dCO₂ = dCO₂dt(CO₂[i], HCO₃[i], CO₃[i], H[i], OH[i], BOH₃[i], BOH₄[i], T, S)
        dHCO₃ = dHCO₃dt(CO₂[i], HCO₃[i], CO₃[i], H[i], OH[i], BOH₃[i], BOH₄[i], T, S)
        dCO₃ = dCO₃dt(CO₂[i], HCO₃[i], CO₃[i], H[i], OH[i], BOH₃[i], BOH₄[i], T, S)
        dH = dHdt(CO₂[i], HCO₃[i], CO₃[i], H[i], OH[i], BOH₃[i], BOH₄[i], T, S)
        dOH = dOHdt(CO₂[i], HCO₃[i], CO₃[i], H[i], OH[i], BOH₃[i], BOH₄[i], T, S)
        dBOH₃ = dBOH₃dt(CO₂[i], HCO₃[i], CO₃[i], H[i], OH[i], BOH₃[i], BOH₄[i], T, S)
        dBOH₄ = - dBOH₃

        dvarsdt1 = [dCO₂, dHCO₃, dCO₃, dH, dOH, dBOH₃, dBOH₄]
        #Uᵐ⁺¹ = Uᵐ + Δt * (γᵐ * Gᵐ + ζᵐ * Gᵐ⁻¹)
        c1 = [CO₂[i], HCO₃[i], CO₃[i], H[i], OH[i], BOH₃[i], BOH₄[i]] #u1
        println("1st RK stage \n")
        @show dvarsdt1
        println("")
        c2 = Array{Float64}(undef, 7)
        for j in 1:size(c1)[1] #first substep
            k1 = dt * (f[1] * dvarsdt1[j])
            c2[j] = c1[j] + k1 #u2
        end 
        @show c2
        println("")

        dCO₂ = dCO₂dt(c2[1], c2[2], c2[3], c2[4], c2[5], c2[6], c2[7], T, S)
        dHCO₃ = dHCO₃dt(c2[1], c2[2], c2[3], c2[4], c2[5], c2[6], c2[7], T, S)
        dCO₃ = dCO₃dt(c2[1], c2[2], c2[3], c2[4], c2[5], c2[6], c2[7], T, S)
        dH = dHdt(c2[1], c2[2], c2[3], c2[4], c2[5], c2[6], c2[7], T, S)
        dOH = dOHdt(c2[1], c2[2], c2[3], c2[4], c2[5], c2[6], c2[7], T, S)
        dBOH₃ = dBOH₃dt(c2[1], c2[2], c2[3], c2[4], c2[5], c2[6], c2[7], T, S)
        dBOH₄ = - dBOH₃

        println("2nd RK stage \n")
        dvarsdt2 = [dCO₂, dHCO₃, dCO₃, dH, dOH, dBOH₃, dBOH₄]
        @show dvarsdt2
        println("")
        c3 = Array{Float64}(undef, 7)
        for j in 1:size(c1)[1] #second substep
            k2 = dt * (f[2] * dvarsdt2[j] + b[2] * dvarsdt1[j])
            c3[j] = c2[j] + k2 #u3
        end 
        @show c3
        println("")
        dCO₂ = dCO₂dt(c3[1], c3[2], c3[3], c3[4], c3[5], c3[6], c3[7], T, S)
        dHCO₃ = dHCO₃dt(c3[1], c3[2], c3[3], c3[4], c3[5], c3[6], c3[7], T, S)
        dCO₃ = dCO₃dt(c3[1], c3[2], c3[3], c3[4], c3[5], c3[6], c3[7], T, S)
        dH = dHdt(c3[1], c3[2], c3[3], c3[4], c3[5], c3[6], c3[7], T, S)
        dOH = dOHdt(c3[1], c3[2], c3[3], c3[4], c3[5], c3[6], c3[7], T, S)
        dBOH₃ = dBOH₃dt(c3[1], c3[2], c3[3], c3[4], c3[5], c3[6], c3[7], T, S)
        dBOH₄ = - dBOH₃
        println("3rd RK stage \n")
        dvarsdt3 = [dCO₂, dHCO₃, dCO₃, dH, dOH, dBOH₃, dBOH₄]
        @show dvarsdt3
        println("")
        c4 = Array{Float64}(undef, 7)
        for j in 1:size(c1)[1] #third substep
            k3 = dt * (f[3] * dvarsdt3[j] + b[3] * dvarsdt2[j])
            c4[j] = c3[j] + k3 #u4
        end
        #update variables
        CO₂[i+1] = c4[1]
        HCO₃[i+1] = c4[2]
        CO₃[i+1] = c4[3]
        H[i+1] = c4[4]
        OH[i+1] = c4[5]
        BOH₃[i+1] = c4[6]
        BOH₄[i+1] = c4[7]
    end
elseif num_method == "euler"
    for i in 1:nt-1
        println("\nTime: ", time[i], " seconds")
        println("CO₂: ", CO₂[i], " mol/m³")
        println("HCO₃: ", HCO₃[i], " mol/m³")
        println("CO₃: ", CO₃[i], " mol/m³")
        println("H: ", H[i], " mol/m³")
        println("OH: ", OH[i], " mol/m³")
        println("BOH₃: ", BOH₃[i], " mol/m³")
        println("BOH₄: ", BOH₄[i], " mol/m³")
        println("")

        dCO₂ = dCO₂dt(CO₂[i], HCO₃[i], CO₃[i], H[i], OH[i], BOH₃[i], BOH₄[i], T, S)
        dHCO₃ = dHCO₃dt(CO₂[i], HCO₃[i], CO₃[i], H[i], OH[i], BOH₃[i], BOH₄[i], T, S)
        dCO₃ = dCO₃dt(CO₂[i], HCO₃[i], CO₃[i], H[i], OH[i], BOH₃[i], BOH₄[i], T, S)
        dH = dHdt(CO₂[i], HCO₃[i], CO₃[i], H[i], OH[i], BOH₃[i], BOH₄[i], T, S)
        dOH = dOHdt(CO₂[i], HCO₃[i], CO₃[i], H[i], OH[i], BOH₃[i], BOH₄[i], T, S)
        dBOH₃ = dBOH₃dt(CO₂[i], HCO₃[i], CO₃[i], H[i], OH[i], BOH₃[i], BOH₄[i], T, S)
        dBOH₄ = - dBOH₃

        dvarsdt1 = [dCO₂, dHCO₃, dCO₃, dH, dOH, dBOH₃, dBOH₄]
        println("Euler step \n")
        @show dvarsdt1
        #update variables
        CO₂[i+1] = CO₂[i] + dt * dCO₂
        HCO₃[i+1] = HCO₃[i]
        CO₃[i+1] = CO₃[i] + dt * dCO₃
        H[i+1] = H[i] + dt * dH
        OH[i+1] = OH[i] + dt * dOH
        BOH₃[i+1] = BOH₃[i] + dt * dBOH₃
        BOH₄[i+1] = BOH₄[i] + dt * dBOH₄
    end
end

println("Time: ", time[nt], " seconds")
println("BOH₃: ", BOH₃[nt], " mol/m³")
println("BOH₄: ", BOH₄[nt], " mol/m³")
println("CO₂: ", CO₂[nt], " mol/m³")
println("CO₃: ", CO₃[nt], " mol/m³")
println("H: ", H[nt], " mol/m³")
println("HCO₃: ", HCO₃[nt], " mol/m³")
println("OH: ", OH[nt], " mol/m³")
println("T: ", T[nt], " °C")
println("S: ", S[nt], " g/kg")
println("")

"using OceanBioME, Oceananigans, Oceananigans.Units
include('cc.jl')
using .CC #: CarbonateChemistry #local module
mutable struct Params
    Nx::Int         # number of points in each of x direction
    Ny::Int         # number of points in each of c1 direction
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

biogeochemistry = CarbonateChemistry(; grid, scale_negatives = true)

model = BoxModel(; biogeochemistry, clock)

set!(model, BOH₃ = 2.97e-4, BOH₄ = 1.19e-4, CO₂ = 7.57e-6, CO₃ = 3.15e-4, H = 6.31e-9, HCO₃ = 1.67e-3, OH = 9.6e-6, T=25, S = 35)

simulation = Simulation(model, Δt=30.0, stop_time = 1hour) #stop_time = 96hours,
@show simulation

BOH₃ = model.fields.BOH₃
BOH₄ = model.fields.BOH₄
CO₂ = model.fields.CO₂
CO₃ = model.fields.CO₃
H = model.fields.H 
HCO₃ = model.fields.HCO₃
OH = model.fields.OH

simulation.output_writers[:fields] = JLD2Writer(model, (; BOH₃, BOH₄, CO₂, CO₃, H, HCO₃, OH),
                                                      schedule = TimeInterval(output_interval),
                                                      filename = 'outputs/box_model.jld2', #$(rank)
                                                      overwrite_existing = true,
                                                      init = save_IC!)
run!(simulation)
"