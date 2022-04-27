@testset "timestep.jl" begin
    @info "Testing timestep.jl"

    timestep = FixedTimestep(0.3)
    @test timestep() == 0.3
    @test_throws ErrorException FixedTimestep(-1)

    timestep = RandomTimestep(Exponential(0.5))
    @test timestep() > 0
    @test timestep() != timestep()

    timestep = DeterministicTrain([1,2,3])
    @test timestep() == 1
    @test timestep() == 2
    @test timestep() == 3
    @test isnothing(timestep())

    @test_throws ErrorException DeterministicTrain([1, 1, -1])
    @test_throws ErrorException DeterministicTrain([1, 0])
end
