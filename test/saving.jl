CUDA.functional() && @testset "Saving Recording using JLD" begin
    N = 10
    p = 0.85
    q = 1.0
    sigma = 0.2
    tau = 0.2

    sim = NestedFilterSimulation(
        N, p, q, sigma, tau,
        1:20,
        LinRange(0.05,0.95,25),
        LinRange(0.1,2,25),
        LinRange(0.05,1,25),
        LinRange(0.05,1,25),
        1024, 256, 12,
        timestep = FixedTimestep(0.35)
    )

    function f1(sim, time)
        Nind = Array(sim.fstate.model.Nind)
        Nrng = Array(sim.fstate.model.Nrng)
        N_posterior = zeros(length(Nrng))
        for j in 1:length(Nrng)
            N_posterior[j] = count(i->(i==j),Nind)
        end
        entropy_N = entropy(N_posterior/sum(N_posterior))

        pind = Array(sim.fstate.model.pind)
        prng = Array(sim.fstate.model.prng)
        p_posterior = zeros(length(prng))
        for j in 1:length(prng)
            p_posterior[j] = count(i->(i==j),pind)
        end
        entropy_p = entropy(p_posterior/sum(p_posterior))

        qind = Array(sim.fstate.model.qind)
        qrng = Array(sim.fstate.model.qrng)
        q_posterior = zeros(length(qrng))
        for j in 1:length(qrng)
            q_posterior[j] = count(i->(i==j),qind)
        end
        entropy_q = entropy(q_posterior/sum(q_posterior))

        σind = Array(sim.fstate.model.σind)
        σrng = Array(sim.fstate.model.σrng)
        σ_posterior = zeros(length(σrng))
        for j in 1:length(σrng)
            σ_posterior[j] = count(i->(i==j),σind)
        end
        entropy_σ = entropy(σ_posterior/sum(σ_posterior))

        τind = Array(sim.fstate.model.τind)
        τrng = Array(sim.fstate.model.τrng)
        τ_posterior = zeros(length(τrng))
        for j in 1:length(τrng)
            τ_posterior[j] = count(i->(i==j),τind)
        end
        entropy_τ = entropy(τ_posterior/sum(τ_posterior))

        map = MAP(sim.fstate.model)
        map_N = map.N
        map_p = map.p
        map_q = map.q
        map_σ = map.σ
        map_τ = map.τ

        return entropy_N, entropy_p, entropy_q, entropy_σ, entropy_τ, map_N, map_p, map_q, map_σ, map_τ, sim.times, time.time
    end

    function f2(data)
        save(string("_testJLDfile.jld"), "data", data)
    end

    rec = Recording(f1, f2, sim)

    sim.hstate.n .= N
    times, epsps = run!(sim, T = 2, recording = Recording(f1, f2, sim))

    @test isfile("_testJLDfile.jld")

    rm("_testJLDfile.jld")
end
