@testset "propagate_emit!" begin
    m_out  = 16
    m_in   = 16

    state = BinomialState(128, m_out, m_in, :cpu)
    model = BinomialModel(128, m_out, :cpu)
    @test_throws ErrorException propagate_emit!(state, model)

    state = ScalarBinomialState(10, 2)
    model = ScalarBinomialModel(10, 0.85, 1.0, 0.2, 0.2)
    @test propagate_emit!(state, model).dt > 0
    @test propagate_emit!(state, model, dt = 0.3).dt == 0.3
    @test propagate_emit!(state, model, λ = 0.5).dt > 0
end
