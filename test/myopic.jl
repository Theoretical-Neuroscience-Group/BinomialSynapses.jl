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

        # test on GPU
        state = BinomialState(10, m_out, m_in)
        model = BinomialModel(10, m_out)
        fstate = NestedParticleState(state, model)

        newstate = _repeat(fstate, m_dts)

        for i in 1:m_dts, j in 1:m_out, k in 1:m_in
            @test Array(newstate.state.n)[i,j,k] == Array(state.n)[j,k]
            @test Array(newstate.state.k)[i,j,k] == Array(state.k)[j,k]
            @test Array(newstate.model.N)[i,j]   == Array(model.N)[j]
            @test Array(newstate.model.p)[i,j]   == Array(model.p)[j]
            @test Array(newstate.model.q)[i,j]   == Array(model.q)[j]
            @test Array(newstate.model.τ)[i,j]   == Array(model.τ)[j]
            @test Array(newstate.model.σ)[i,j]   == Array(model.σ)[j]
        end
    end

    @testset "_entropy" begin
        using BinomialSynapses: _entropy

        Nrng = 1:5
        prng = 0.1:0.2:0.9
        qrng = 0.1:0.2:0.9
        σrng = 0.5:0.5:2.5
        τrng = 0.1:0.1:0.5

        Nind = [
            1 1 1 1 1 1;
            1 1 1 2 2 2;
            1 1 2 2 2 2;
            1 1 2 2 3 3;
        ]

        pind = qind = σind = τind = Nind

        model = BinomialGridModel(
            Nind, pind, qind, σind, τind,
            Nrng, prng, qrng, σrng, τrng
        )
        
        policy = Myopic([1.])

        @test _entropy(model, policy) ≈ [0., log(2), 1/3*log(3)+2/3*log(3/2), log(3)]

        # test parameters on GPU

        Nrng = CuArray(Int.(Nrng))
        prng = CuArray(Float32.(prng))
        qrng = CuArray(Float32.(qrng))
        σrng = CuArray(Float32.(σrng))
        τrng = CuArray(Float32.(τrng))

        Nind = cu(Nind)
        pind = qind = σind = τind = Nind

        model = BinomialGridModel(
            cu(Nind), cu(pind), cu(qind), cu(σind), cu(τind),
            cu(Nrng), cu(prng), cu(qrng), cu(σrng), cu(τrng)
        )
        
        policy = Myopic([1.])

        @test _entropy(model, policy) ≈ [0., log(2), 1/3*log(3)+2/3*log(3/2), log(3)]
    end
end
