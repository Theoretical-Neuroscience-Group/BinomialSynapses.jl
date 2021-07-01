function _tauentropy(model::BinomialGridModel) 
    # CPU algorithm: move index arrays to CPU
    τind = Array(model.τind)

    counts = Dict{Int, Int}()
    @inbounds for i in 1:length(τind)
        iτ = τind[i]
        key = iτ
        counts[key] = get!(counts, key, 0) + 1
    end

    entropy = 0.
    @inbounds for count in values(counts)
        p = count / length(τind)
        entropy -= p * log(p)
    end
    
    return entropy
end

@testset "debugging" begin
    println("             > DEBUGGING")
    @testset "initial τ entropy" begin
        N = 10
        p = 0.85
        q = 1.0
        σ = 0.2
        τ = 0.2

        n = 1000
        avg_entropy = 0.
        for i in 1:n
            sim = NestedFilterSimulation(
                    N, p, q, σ, τ,
                    1:20,
                    LinRange(0.00, 1.00, 45),
                    LinRange(0.00, 2.00, 45),
                    LinRange(0.05, 2.00, 45),
                    LinRange(0.05, 2.00, 45),
                    2048, 512, 12
                )

            run!(sim, T = 1)

            avg_entropy += _tauentropy(sim.fstate.model)
        end
        avg_entropy /= n
        @test 3.6 < avg_entropy < 3.8
    end
end
