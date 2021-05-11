@testset "timestep.jl" begin
    println("             > timestep.jl")

    timestep = FixedTimestep(0.3)
    @test get_step(timestep) == 0.3

    timestep = RandomTimestep(Exponential(0.5))
    @test get_step(timestep) > 0
    @test get_step(timestep) != get_step(timestep)
end
