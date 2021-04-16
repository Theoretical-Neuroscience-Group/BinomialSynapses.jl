@testset "resample!" begin

    N, K = 10, 10

    wv = Array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    a = CuArray{Int64}(1:N)
    x = CuArray{Int64}(zeros(Int64, K));

    cu_alias_sample!(a,wv, x)

    # Resampling should only pick the second index (the only one with a non-
    # zero likelihood)
    @test all(x .== 2)

end
