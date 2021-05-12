@testset "myopic.jl" begin
    println("             > myopic.jl")
    @testset "_repeat" begin
        using BinomialSynapses: _repeat
        m_out = 3
        m_in = 4
        m_dts = 5
        state = BinomialState(10, m_out, m_in, :cpu)
        model = BinomialModel(10, m_out, :cpu)
        fstate = NestedParticleState(state, model)

        newstate = _repeat(fstate, m_dts)

        for i in 1:m_dts, j in 1:m_out, k in 1:m_in
            @test newstate.state.n[i,j,k] == state.n[j,k]
            @test newstate.state.k[i,j,k] == state.k[j,k]
            @test newstate.model.N[i,j]   == model.N[j]
            @test newstate.model.p[i,j]   == model.p[j]
            @test newstate.model.q[i,j]   == model.q[j]
            @test newstate.model.τ[i,j]   == model.τ[j]
            @test newstate.model.σ[i,j]   == model.σ[j]
        end
    end
end
