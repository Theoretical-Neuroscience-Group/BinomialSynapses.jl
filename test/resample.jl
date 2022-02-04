@testset "resample.jl" begin
    println("             > resample.jl")
    @testset "indices!" begin
        if RUN_BENCHMARKS
            @testset "benchmark" begin
                println("")
                println("Benchmarking function indices!, 1D: should take about 90μs")
                v = CUDA.rand(2048)
                display(@benchmark CUDA.@sync indices!($v))
                println("")
                println("")
                println("Benchmarking function indices!, 1D: should take about 550μs")
                v = CUDA.rand(2048, 512)
                display(@benchmark CUDA.@sync indices!($v))
                println("")
                println("")
            end
        end
        @testset "1D" begin
            v = cu([1f0, 1f4, 1f0, 0f0])
            vold = copy(v)
            u, idx = indices!(v)

            @test size(u) == ()
            @test size(idx) == (4,)
            @test u ≈ sum(vold)

            idx = Array(idx)

            @test idx == [2, 2, 2, 2]

            @testset "v of dim ($M_out,)" for M_out in [4, 8, 16, 32, 64, 128, 256, 512, 1024]
                v = CUDA.rand(M_out)
                vold = copy(v)
                u, idx = indices!(v)

                @test size(u) == ()
                @test size(idx) == (M_out,)
                @test u ≈ sum(vold)

                idx = Array(idx)
                @test issorted(idx)
            end
        end
        @testset "2D" begin
            v = cu([1f0 1f4 1f0 0f0;
                    1f0 1f0 1f4 1f0])
            vold = copy(v)
            u, idx = indices!(v)

            @test size(u) == (2,)
            @test size(idx) == (2, 4)
            @test v ≈ cumsum(vold, dims = 2)
            @test u ≈ sum(vold, dims = 2)

            idx = Array(idx)

            @test idx == [2 2 2 2;
                          3 3 3 3]

            @testset "v of dim ($M_out, $M_in)" for M_out in [4, 16, 64, 256, 1024], 
                M_in in [4, 16, 64, 256, 1024]
                v = CUDA.rand(M_out, M_in)
                vold = copy(v)
                u, idx = indices!(v)

                @test size(u) == (M_out,)
                @test size(idx) == (M_out, M_in)
                @test v ≈ cumsum(vold, dims = 2)
                @test u ≈ sum(vold, dims = 2)

                idx = Array(idx)
                for idxrow in eachrow(idx)
                    @test issorted(idxrow)
                end
            end
        end
        @testset "3D" begin
            v = ones(Float32, 2, 3, 4)
            trueidx = [2 4 3; 1 3 2]
            for i in 1:2, j in 1:3
                v[i, j, trueidx[i, j]] = 1f4
            end
            v = cu(v)
            vold = copy(v)
            u, idx = indices!(v)

            @test size(u) == (2, 3)
            @test size(idx) == (2, 3, 4)
            @test v ≈ cumsum(vold, dims = 3)
            @test u ≈ sum(vold, dims = 3)

            idx = Array(idx)

            for i in 1:2, j in 1:3, k in 1:4
                @test idx[i, j, k] == trueidx[i, j]
            end

            @testset "v of dim ($M_dt, $M_out, $M_in)" for M_dt in [4, 16], M_out in [4, 64, 1024], M_in in [4, 64, 1024]
                v = CUDA.rand(M_dt, M_out, M_in)
                vold = copy(v)
                u, idx = indices!(v)

                @test size(u) == (M_dt, M_out)
                @test size(idx) == (M_dt, M_out, M_in)
                @test v ≈ cumsum(vold, dims = 3)
                @test u ≈ sum(vold, dims = 3)

                idx = Array(idx)
                for i in 1:M_dt, j in 1:M_out
                    @test issorted(idx[i, j, :])
                end
            end
        end
    end

    @testset "resample!" begin
        @testset "1D, 1D" begin
            A = CUDA.rand(8)
            B = CUDA.zeros(8)
            idx = cu(rand(1:8, 8))

            resample!(A, B, idx)

            A = Array(A)
            B = Array(B)
            idx = Array(idx)

            for i in 1:length(A)
                @test B[i] == A[idx[i]]
            end
        end
        @testset "2D, 1D" begin
            A = CUDA.rand(8, 8)
            B = CUDA.zeros(8, 8)
            idx = cu(rand(1:8, 8))

            resample!(A, B, idx)

            A = Array(A)
            B = Array(B)
            idx = Array(idx)

            for i in 1:size(A, 1), j in 1:size(A, 2)
                @test B[i, j] == A[idx[i], j]
            end
        end
        @testset "2D, 2D" begin
            A = CUDA.rand(8, 8)
            B = CUDA.zeros(8, 8)
            idx = cu(rand(1:8, 8, 8))

            resample!(A, B, idx)

            A = Array(A)
            B = Array(B)
            idx = Array(idx)

            for i in 1:size(A, 1), j in 1:size(A, 2)
                @test B[i, j] == A[i, idx[i, j]]
            end
        end
        @testset "3D, 1D" begin
            A = CUDA.rand(8, 8, 8)
            B = CUDA.zeros(8, 8, 8)
            idx = cu(rand(1:8, 8))

            resample!(A, B, idx)

            A = Array(A)
            B = Array(B)
            idx = Array(idx)

            for i in 1:size(A, 1), j in 1:size(A, 2), k in 1:size(A, 3)
                @test B[i, j, k] == A[idx[i], j, k]
            end
        end
        @testset "3D, 2D" begin
            A = CUDA.rand(8, 8, 8)
            B = CUDA.zeros(8, 8, 8)
            idx = cu(rand(1:8, 8, 8))

            resample!(A, B, idx)

            A = Array(A)
            B = Array(B)
            idx = Array(idx)

            for i in 1:size(A, 1), j in 1:size(A, 2), k in 1:size(A, 3)
                @test B[i, j, k] == A[i, idx[i, j], k]
            end
        end
    end

    @testset "outer_resample!" begin
        if RUN_BENCHMARKS
            m_out = 1024
            m_in  = 1024
            state = BinomialState(128, m_out, m_in)
            model = BinomialGridModel(
                m_out,
                1:5,
                LinRange(0.05,0.95,5),
                LinRange(0.1,2,5),
                LinRange(0.05,2,5),
                LinRange(0.05,2,5)
            )

            u = CUDA.rand(m_out)

            println("")
            println("Benchmarking function outer_resample!: should take about 300μs")
            display(@benchmark outer_resample!($state, $model, $u))
            println("")
            println("")
        end
    end
end
