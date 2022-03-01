@testset "timestep.jl" begin
    @info "Testing timestep.jl"

    timestep = FixedTimestep(0.3)
    @test get_step(timestep) == 0.3
    @test_throws ErrorException FixedTimestep(-1)

    timestep = RandomTimestep(Exponential(0.5))
    @test get_step(timestep) > 0
    @test get_step(timestep) != get_step(timestep)

    timestep = DeterministicTrain([1,2,3])
    @test get_step(timestep) == 1
    @test get_step(timestep) == 2
    @test get_step(timestep) == 3
    @test isnothing(get_step(timestep))

    @test_throws ErrorException DeterministicTrain([1, 1, -1])
    @test_throws ErrorException DeterministicTrain([1, 0])
end
