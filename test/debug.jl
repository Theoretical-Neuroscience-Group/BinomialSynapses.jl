@testset "debug convergence speed" begin
    function run_until_entropy_below!(
        sim::NestedFilterSimulation; 
        threshold::Real
    )
        if length(sim.times) == 0
            initialize!(sim)
        end
        t = 0
        while true
            t += 1
            begin
                propagate!(sim)
            end
            e = sim.fstate.model.τ |> collect |> proportionmap |> values |> entropy
            @show e
            if e < threshold || t > 10000
                return t
            end
        end
    end

    sim = NestedFilterSimulation(
        10, 0.85, 1.0, 0.2, 0.2,
        1:20,
        LinRange(0.00, 1.00, 45),
        LinRange(0.00, 2.00, 45),
        LinRange(0.05, 2.00, 45),
        LinRange(0.05, 2.00, 45),
        2048, 512, 12,
        timestep = RandomTimestep(Exponential(0.121))                 
    )
    
    number_of_timesteps = run_until_entropy_below!(sim, threshold = 1.9)
    @show number_of_timesteps
end
