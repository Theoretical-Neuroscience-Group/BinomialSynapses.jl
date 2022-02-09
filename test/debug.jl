@testset "debug convergence speed" begin
    function run_until_entropy_below(
        threshold::Real
    )
        sim = NestedFilterSimulation(
            10, 0.85, 1.0, 0.2, 0.2,
            1:20,
            LinRange(0.00, 1.00, 45),
            LinRange(0.00, 2.00, 45),
            LinRange(0.05, 2.00, 45),
            LinRange(0.05, 2.00, 45),
            2048, 512, 12             
        )
        times = zeros(0)
        epsps = zeros(0)
        time = 0.
        delta = 0.

        t = 0
        while true
            t += 1
            obs = propagate!(sim, λ = 0.121)
            push!(times, time += obs.dt)
            push!(epsps, obs.EPSP)
            e = sim.fstate.model.τ |> collect |> proportionmap |> values |> entropy
            if e < threshold || t > 1000
                return t
            end
        end

        # initialize!(sim)
        # t = 0
        # while true
        #     t += 1
        #     propagate!(sim)
        #     e = sim.fstate.model.τ |> collect |> proportionmap |> values |> entropy
        #     if e < threshold || t > 1000
        #         return t
        #     end
        # end
    end

    number_of_timesteps = 0.
    N = 200
    for i in 1:N
        nn = run_until_entropy_below(1.9)
        @show nn
        number_of_timesteps += nn
    end
    number_of_timesteps /= N
    @show number_of_timesteps
end
