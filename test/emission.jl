@testset "emission.jl" begin
    println("             > emission.jl")
    m_out  = 16
    m_in   = 16

    state = BinomialState(128, m_out, m_in, :cpu)
    model = BinomialModel(128, m_out, :cpu)
    @test_throws ErrorException emit(state, model, 0.3)

    state = ScalarBinomialState(10, 2)
    model = ScalarBinomialModel(10, 0.85, 1.0, 0.2, 0.2)
    @test emit(state, model, 0.3).dt == 0.3
    @test emit(state, model, rand(Exponential(0.5))).dt > 0
end
