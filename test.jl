using OceanBioME, Test, CUDA, Oceananigans, JLD2, Oceananigans.Units, Documenter

architecture = CUDA.has_cuda() ? GPU() : CPU()

using OceanBioME: CarbonateChemistry
using Oceananigans

function test_CarbonateChemistry(grid, sinking, open_bottom)
    
    if sinking
        model = NonhydrostaticModel(;grid,
                                     biogeochemistry = CarbonateChemistry(;grid, open_bottom))
    else
        model = NonhydrostaticModel(;grid,
                                     biogeochemistry = CarbonateChemistry(;grid, sinking_speeds = NamedTuple()))
    end

    # correct tracers and auxiliary fields have been setup, and order has not changed
    required_tracers = (:CO2, :HCO3, :CO3, :OH, :BOH3, :BOH4, :T)

    @test Oceananigans.Biogeochemistry.required_biogeochemical_tracers(model.biogeochemistry) == required_tracers
    @test all(tracer ∈ keys(model.tracers) for tracer in required_tracers)
    # checks model works with zero values
    time_step!(model, 1.0e-7)

    # and that they all return zero
    @test all([all(Array(interior(values)) .== 0) for values in values(model.tracers)]) 

    # mass conservation
    #set!(model, BOH3 = 2.97e2, BOH4 = 1.19e2, CO2 = 7.57e0 * 1.0e3, CO3 = 3.15e2, HCO3 = 1.67e3, OH = 9.6e0, T=25.0) 
    set!(model, BOH3 = rand(), BOH4 = rand(), CO2 = rand(), CO3 = rand(), HCO3 = rand(), OH = rand(), T=25.0) 

    ΣN₀ = sum(Array(interior(model.tracers.BOH3))) + 
          sum(Array(interior(model.tracers.BOH4))) + 
          sum(Array(interior(model.tracers.CO2))) + 
          sum(Array(interior(model.tracers.CO3))) + 
          sum(Array(interior(model.tracers.HCO3))) + 
          sum(Array(interior(model.tracers.OH)))

    for n in 1:1000
        time_step!(model, 1.0e-7)
    end

    ΣN₁ = sum(Array(interior(model.tracers.BOH3))) + 
          sum(Array(interior(model.tracers.BOH4))) + 
          sum(Array(interior(model.tracers.CO2))) + 
          sum(Array(interior(model.tracers.CO3))) + 
          sum(Array(interior(model.tracers.HCO3))) + 
          sum(Array(interior(model.tracers.OH)))

    @test ΣN₀ ≈ ΣN₁ # guess this should actually fail with a high enough accuracy when sinking is on with an open bottom

    return nothing
end

grid = RectilinearGrid(architecture; size=(32, 32, 32), extent=(1, 1, 2))

for sinking = (false, true), open_bottom = (false, true)
    if !(sinking && open_bottom) # no sinking is the same with and without open bottom
        @info "Testing on $(typeof(architecture)) with sinking $(sinking ? :✅ : :❌), open bottom $(open_bottom ? :✅ : :❌))"
        @testset "CarbonateChemistry $architecture, $sinking, $open_bottom" begin
            test_CarbonateChemistry(grid, sinking, open_bottom)
        end
    end
end

@testset "Float32 CarbonateChemistry" begin
    grid = RectilinearGrid(architecture, Float32; size=(32, 32, 32), extent=(10, 10, 200))
    bgc = CarbonateChemistry(; grid)
    ubgc = bgc.underlying_biogeochemistry
    @test ubgc.A1 isa Float32
    @test ubgc.E1 isa Float32
    @test ubgc.A7 isa Float32
    @test ubgc.E7 isa Float32
    @test ubgc.A8 isa Float32
    @test ubgc.E8 isa Float32
    @test ubgc.alpha3 isa Float32
    @test ubgc.alpha4 isa Float32
    @test ubgc.alpha5 isa Float32
end
