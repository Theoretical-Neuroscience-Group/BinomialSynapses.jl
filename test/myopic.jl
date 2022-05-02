@testset "myopic.jl" begin
    @info "Testing myopic.jl"
    @testset "_repeat" begin
        m_out = 3
        m_in = 4
        m_dts = 5
        @testset "Device = $device" for device in DEVICES
            using BinomialSynapses: _repeat
        
            state = BinomialState(10, m_out, m_in, device = device)
            model = BinomialModel(10, m_out, device = device)
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
    end

    @testset "_entropy: Myopic" begin
        using BinomialSynapses: _entropy

        Nrng = 1:5
        prng = 0.1:0.2:0.9
        qrng = 0.1:0.2:0.9
        σrng = 0.5:0.5:2.5
        τrng = 0.1:0.1:0.5

        @testset "CPU" begin
            Nind = [
                1 1 1 2 2 2;
                1 1 2 2 2 2;
                1 1 1 1 1 1;
                1 1 2 2 3 3;
            ]

            pind = qind = σind = τind = Nind

            model = BinomialGridModel(
                Nind, pind, qind, σind, τind,
                Nrng, prng, qrng, σrng, τrng
            )
            
            policy = Myopic([1.])
            obs = BinomialObservation(zeros(4), [0.1, 0.2, 0.3, 0.4])

            @test _entropy(model, obs, policy) ≈ 0.3
        end

        CUDA.functional() && @testset "GPU" begin
            Nrng = CuArray(Int.(Nrng))
            prng = CuArray(Float32.(prng))
            qrng = CuArray(Float32.(qrng))
            σrng = CuArray(Float32.(σrng))
            τrng = CuArray(Float32.(τrng))

            Nind = cu([
                1 1 1 2 2 2;
                1 1 1 1 1 1;
                1 1 2 2 2 2;
                1 1 2 2 3 3;
            ])
            pind = qind = σind = τind = Nind

            model = BinomialGridModel(
                cu(Nind), cu(pind), cu(qind), cu(σind), cu(τind),
                cu(Nrng), cu(prng), cu(qrng), cu(σrng), cu(τrng)
            )
            
            policy = Myopic([1.])
            obs = BinomialObservation(zeros(4), [0.1, 0.2, 0.3, 0.4])

            @test _entropy(model, obs, policy) ≈ 0.2
        end
    end

    @testset "_entropy: MyopicFast" begin
        using BinomialSynapses: _entropy
        
        Nrng = 1:5
        prng = 0.1:0.2:0.9
        qrng = 0.1:0.2:0.9
        σrng = 0.5:0.5:2.5
        τrng = 0.1:0.1:0.5
        
        @testset "CPU" begin
            Nind = [2, 1, 1, 2, 3, 2, 2]
            pind = qind = σind = τind = Nind

            model = BinomialGridModel(
                Nind, pind, qind, σind, τind,
                Nrng, prng, qrng, σrng, τrng
            )
            
            policy = MyopicFast([1.])
            obs = BinomialObservation(zeros(4), [0.1, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3])

            @test _entropy(model, obs, policy) ≈ 0.3
        end

        CUDA.functional() && @testset "GPU" begin
            Nrng = CuArray(Int.(Nrng))
            prng = CuArray(Float32.(prng))
            qrng = CuArray(Float32.(qrng))
            σrng = CuArray(Float32.(σrng))
            τrng = CuArray(Float32.(τrng))

            Nind = cu([2, 1, 1, 2, 3, 2, 3, 3])
            pind = qind = σind = τind = Nind

            model = BinomialGridModel(
                cu(Nind), cu(pind), cu(qind), cu(σind), cu(τind),
                cu(Nrng), cu(prng), cu(qrng), cu(σrng), cu(τrng)
            )
            
            policy = Myopic([1.])
            obs = BinomialObservation(zeros(4), [0.1, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3])

            @test _entropy(model, obs, policy) ≈ 0.1
        end
    end

#######################################################################################

    @testset "_tauentropy: Myopic" begin
        using BinomialSynapses: _tauentropy

        Nrng = 1:5
        prng = 0.1:0.2:0.9
        qrng = 0.1:0.2:0.9
        σrng = 0.5:0.5:2.5
        τrng = 0.1:0.1:0.5

        @testset "CPU" begin
            Nind = [
                1 1 1 2 2 2;
                1 1 2 2 2 2;
                1 1 1 1 1 1;
                1 1 2 2 3 3;
            ]

            pind = qind = σind = τind = Nind

            model = BinomialGridModel(
                Nind, pind, qind, σind, τind,
                Nrng, prng, qrng, σrng, τrng
            )
            
            policy = Myopic([1.])
            obs = BinomialObservation(zeros(4), [0.1, 0.2, 0.3, 0.4])

            @test _tauentropy(model, obs, policy) ≈ 0.3
        end

        CUDA.functional() && @testset "GPU" begin
            Nrng = CuArray(Int.(Nrng))
            prng = CuArray(Float32.(prng))
            qrng = CuArray(Float32.(qrng))
            σrng = CuArray(Float32.(σrng))
            τrng = CuArray(Float32.(τrng))

            Nind = cu([
                1 1 1 2 2 2;
                1 1 1 1 1 1;
                1 1 2 2 2 2;
                1 1 2 2 3 3;
            ])
            pind = qind = σind = τind = Nind

            model = BinomialGridModel(
                cu(Nind), cu(pind), cu(qind), cu(σind), cu(τind),
                cu(Nrng), cu(prng), cu(qrng), cu(σrng), cu(τrng)
            )
            
            policy = Myopic([1.])
            obs = BinomialObservation(zeros(4), [0.1, 0.2, 0.3, 0.4])

            @test _tauentropy(model, obs, policy) ≈ 0.2
        end
    end

    @testset "_tauentropy: MyopicFast" begin
        using BinomialSynapses: _tauentropy

        Nrng = 1:5
        prng = 0.1:0.2:0.9
        qrng = 0.1:0.2:0.9
        σrng = 0.5:0.5:2.5
        τrng = 0.1:0.1:0.5

        @testset "CPU" begin
            Nind = [2, 1, 1, 2, 3, 2, 2]
            pind = qind = σind = τind = Nind

            model = BinomialGridModel(
                Nind, pind, qind, σind, τind,
                Nrng, prng, qrng, σrng, τrng
            )
            
            policy = MyopicFast([1.])
            obs = BinomialObservation(zeros(4), [0.1, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3])

            @test _tauentropy(model, obs, policy) ≈ 0.3
        end

        CUDA.functional() && @testset "GPU" begin
            Nrng = CuArray(Int.(Nrng))
            prng = CuArray(Float32.(prng))
            qrng = CuArray(Float32.(qrng))
            σrng = CuArray(Float32.(σrng))
            τrng = CuArray(Float32.(τrng))

            Nind = cu([2, 1, 1, 2, 3, 2, 3, 3])
            pind = qind = σind = τind = Nind

            model = BinomialGridModel(
                cu(Nind), cu(pind), cu(qind), cu(σind), cu(τind),
                cu(Nrng), cu(prng), cu(qrng), cu(σrng), cu(τrng)
            )
            
            policy = Myopic([1.])
            obs = BinomialObservation(zeros(4), [0.1, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3])

            @test _tauentropy(model, obs, policy) ≈ 0.1
        end
    end

    @testset "_diffentropy: Myopic" begin
        using BinomialSynapses: _diffentropy

        Nrng = 1:5
        prng = 0.1:0.2:0.9
        qrng = 0.1:0.2:0.9
        σrng = 0.5:0.5:2.5
        τrng = 0.1:0.1:0.5

        @testset "CPU" begin
            Nind = rand(1:4, 4, 10)
            Nind[3, 1:5] .= 1
            Nind[3, 6:10] .= 2

            pind = qind = σind = τind = Nind

            model = BinomialGridModel(
                Nind, pind, qind, σind, τind,
                Nrng, prng, qrng, σrng, τrng
            )
            
            policy = Myopic([1.])
            obs = BinomialObservation(zeros(4), [0.1, 0.2, 0.3, 0.4])

            @test _diffentropy(model, obs, policy) ≈ 0.3
        end

        CUDA.functional() && @testset "GPU" begin
            Nrng = CuArray(Int.(Nrng))
            prng = CuArray(Float32.(prng))
            qrng = CuArray(Float32.(qrng))
            σrng = CuArray(Float32.(σrng))
            τrng = CuArray(Float32.(τrng))

            Nind = rand(1:4, 4, 10)
            Nind[2, 1:5] .= 1
            Nind[2, 6:10] .= 2
            pind = qind = σind = τind = Nind

            Nind = cu(Nind)

            model = BinomialGridModel(
                cu(Nind), cu(pind), cu(qind), cu(σind), cu(τind),
                cu(Nrng), cu(prng), cu(qrng), cu(σrng), cu(τrng)
            )
            
            policy = Myopic([1.])
            obs = BinomialObservation(zeros(4), [0.1, 0.2, 0.3, 0.4])

            @test _diffentropy(model, obs, policy) ≈ 0.2
        end
    end
end

