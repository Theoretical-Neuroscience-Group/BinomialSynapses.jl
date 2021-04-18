@testset "update!" begin
    M_out    = 5
    Nrng     = CuArray(1:5)
    prng     = CuArray(Float32.(LinRange(0.05,0.95,5)))
    qrng     = CuArray(Float32.(LinRange(0.1,2,5)))
    sigmarng = CuArray(Float32.(LinRange(0.05,2,5)))
    taurng   = CuArray(Float32.(LinRange(0.05,2,5)))
    Nind     = CUDA.ones(Int, M_out)
    pind     = CUDA.ones(Int, M_out)
    qind     = CUDA.ones(Int, M_out)
    sigmaind = CUDA.ones(Int, M_out)
    tauind   = CUDA.ones(Int, M_out)

    model = BinomialGridModel(
        Nind, pind, qind, sigmaind, tauind,
        Nrng, prng, qrng, sigmarng, taurng
    )

    for i in 1:100
        model_old = deepcopy(model)
        jitter!(model, 12)
        # check whether new indices are in the allowed range of indices
        @test all(1 .<= model.Nind     .<= length(Nrng))
        @test all(1 .<= model.pind     .<= length(prng))
        @test all(1 .<= model.qind     .<= length(qrng))
        @test all(1 .<= model.sigmaind .<= length(sigmarng))
        @test all(1 .<= model.tauind   .<= length(taurng))

        # check whether new indices are different by at most one from old indices
        @test all(abs.(model.Nind     .- model_old.Nind)     .<= 1)
        @test all(abs.(model.pind     .- model_old.pind)     .<= 1)
        @test all(abs.(model.qind     .- model_old.qind)     .<= 1)
        @test all(abs.(model.sigmaind .- model_old.sigmaind) .<= 1)
        @test all(abs.(model.tauind   .- model_old.tauind)   .<= 1)
    end

    M_out    = 1024
    Nrng     = CuArray(1:5)
    prng     = CuArray(Float32.(LinRange(0.05,0.95,5)))
    qrng     = CuArray(Float32.(LinRange(0.1,2,5)))
    sigmarng = CuArray(Float32.(LinRange(0.05,2,5)))
    taurng   = CuArray(Float32.(LinRange(0.05,2,5)))
    Nind     = CUDA.ones(Int, M_out)
    pind     = CUDA.ones(Int, M_out)
    qind     = CUDA.ones(Int, M_out)
    sigmaind = CUDA.ones(Int, M_out)
    tauind   = CUDA.ones(Int, M_out)

    model = BinomialGridModel(
        Nind, pind, qind, sigmaind, tauind,
        Nrng, prng, qrng, sigmarng, taurng
    )

    println("")
    println("Benchmarking function jitter!: should take about 80μs")
    display(@benchmark CUDA.@sync jitter!($model, 12))
    println("")
end
