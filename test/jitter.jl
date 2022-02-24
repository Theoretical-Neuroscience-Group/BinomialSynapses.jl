CUDA.functional() && @testset "jitter.jl" begin
    println("             > jitter.jl")
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
        @test all(1 .<= model.Nind .<= length(model.Nrng))
        @test all(1 .<= model.pind .<= length(model.prng))
        @test all(1 .<= model.qind .<= length(model.qrng))
        @test all(1 .<= model.σind .<= length(model.σrng))
        @test all(1 .<= model.τind .<= length(model.τrng))

        # check whether new indices are different by at most one from old indices
        @test all(abs.(model.Nind .- model_old.Nind) .<= 1)
        @test all(abs.(model.pind .- model_old.pind) .<= 1)
        @test all(abs.(model.qind .- model_old.qind) .<= 1)
        @test all(abs.(model.σind .- model_old.σind) .<= 1)
        @test all(abs.(model.τind .- model_old.τind) .<= 1)
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

    if RUN_BENCHMARKS
        println("")
        println("Benchmarking function jitter!: should take about 80μs")
        display(@benchmark CUDA.@sync jitter!($model, 12))
        println("")
        println("")
    end
end
