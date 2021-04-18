@testset "jitter!" begin
    m_out = 5
    model = BinomialGridModel(
        m_out,
        1:5,
        LinRange(0.05,0.95,5),
        LinRange(0.1,2,5),
        LinRange(0.05,2,5),
        LinRange(0.05,2,5)
    )

    for i in 1:100
        model_old = deepcopy(model)
        jitter!(model, 12)
        
        # check whether new indices are in the allowed range of indices
        @test all(1 .<= model.Nind     .<= length(model.Nrng))
        @test all(1 .<= model.pind     .<= length(model.prng))
        @test all(1 .<= model.qind     .<= length(model.qrng))
        @test all(1 .<= model.sigmaind .<= length(model.sigmarng))
        @test all(1 .<= model.tauind   .<= length(model.taurng))

        # check whether new indices are different by at most one from old indices
        @test all(abs.(model.Nind     .- model_old.Nind)     .<= 1)
        @test all(abs.(model.pind     .- model_old.pind)     .<= 1)
        @test all(abs.(model.qind     .- model_old.qind)     .<= 1)
        @test all(abs.(model.sigmaind .- model_old.sigmaind) .<= 1)
        @test all(abs.(model.tauind   .- model_old.tauind)   .<= 1)
    end

    m_out = 1024
    model = BinomialGridModel(
        m_out,
        1:5,
        LinRange(0.05,0.95,5),
        LinRange(0.1,2,5),
        LinRange(0.05,2,5),
        LinRange(0.05,2,5)
    )

    println("")
    println("Benchmarking function jitter!: should take about 80μs")
    display(@benchmark CUDA.@sync jitter!($model, 12))
    println("")
end
