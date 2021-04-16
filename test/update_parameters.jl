@testset "draw_theta!" begin
    M_out = 5
    N_range         = 1:5
    p_range         = LinRange(0.05,0.95,5)
    q_range         = LinRange(0.1,2,5)
    sigma_range     = LinRange(0.05,2,5)
    tau_range       = LinRange(0.05,2,5)
    indexes = CUDA.ones(Int,M_out,5)
    Ns = CUDA.ones(Int,M_out)
    ps = Float32(0.05).*CUDA.ones(M_out)
    qs = Float32(0.1).*CUDA.ones(M_out)
    sigmas = Float32(0.05).*CUDA.ones(M_out)
    taus = Float32(0.05).*CUDA.ones(M_out)

    for i in 1:100
        indexes_old = indexes
        draw_theta!(indexes,N_range,p_range,q_range,sigma_range,tau_range,12,Ns,ps,qs,sigmas,taus)
        @test all(abs.(indexes_old.-indexes) .<= 1)
    end

end
