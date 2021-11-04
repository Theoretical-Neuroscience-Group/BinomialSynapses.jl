struct NestedFilterSimulation{T1, T2, T3, T4, T5, T6, T7}
    hmodel::T1
    filter::T2
    hstate::T3
    fstate::T4
    tsteps::T5
    times::T6
    epsps::T7
end

struct Map{T1, T2, T3, T4, T5}
    map_N::T1
    map_p::T2
    map_q::T3
    map_σ::T4
    map_τ::T5
end

struct Entropies{T1, T2, T3, T4, T5}
    entropy_N::T1
    entropy_p::T2
    entropy_q::T3
    entropy_σ::T4
    entropy_τ::T5
end

struct Results{T1, T2, T3}
    entropies::T1
    map::T2
    time::T3
end

function NestedFilterSimulation(
    N, p, q, σ, τ,
    Nrng, prng, qrng, σrng, τrng,
    m_out, m_in, width;
    timestep::Timestep = RandomTimestep(Exponential(0.121))
)
    hmodel = ScalarBinomialModel(N, p, q, σ, τ)
    filter = NestedParticleFilter(width)
    hstate = ScalarBinomialState(N, 0)
    fstate = NestedParticleState(
                m_out, m_in,
                Nrng, prng, qrng, σrng, τrng
             )
    times = zeros(0)
    epsps = zeros(0)
    return NestedFilterSimulation(hmodel, filter, hstate, fstate, timestep, times, epsps)
end

function m_out(sim::NestedFilterSimulation)
    return size(sim.fstate.state.n, 1)
end

function propagate_hidden!(sim, dt)
    return propagate!(sim.hstate, sim.hmodel, dt)
end

function emit(sim::NestedFilterSimulation, dt)
    return emit(sim.hstate, sim.hmodel, dt)
end

function filter_update!(sim::NestedFilterSimulation, obs)
    return update!(sim.fstate, obs, sim.filter)
end

function initialize!(sim::NestedFilterSimulation)
    dt = 0.
    propagate_hidden!(sim, dt)
    obs = emit(sim, dt)
    filter_update!(sim, obs)
    push!(sim.times, dt)
    push!(sim.epsps, obs.EPSP)
    return sim
end

get_step(sim::NestedFilterSimulation) = get_step(sim.tsteps)

function propagate!(sim::NestedFilterSimulation)
    dt = get_step(sim)
    propagate!(sim, dt)
end

function propagate!(sim::NestedFilterSimulation, dt)
    propagate_hidden!(sim, dt)
    obs = emit(sim, dt)
    filter_update!(sim, obs)
    push!(sim.times, sim.times[end] + dt)
    push!(sim.epsps, obs.EPSP)
    return sim
end

function run!(sim::NestedFilterSimulation; T::Int, plot_each_timestep::Bool = false)
    entropies = Entropies(zeros(T),zeros(T),zeros(T),zeros(T),zeros(T))
    Maps = Map(zeros(T),zeros(T),zeros(T),zeros(T),zeros(T))
            
    results = Results(entropies,Maps,zeros(T))
    
    if length(sim.times) == 0
        initialize!(sim)
    end
    for i in 1:T
        begin
            time = @timed propagate!(sim)
        end
        if plot_each_timestep
            posterior_plot(sim)
        end
        save_results!(results, sim, i, time)
        if i == T
            save(string(Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"]),".jld"), "entropies", results.entropies, "ISI", sim.times, "MAP", results.map)
        end
    end
    return sim.times, sim.epsps
end

function save_results!(results::Results, sim::NestedFilterSimulation, i, time)
    
    results.time[i] = time
    
    Nind = Array(sim.fstate.model.Nind)
    Nrng = Array(sim.fstate.model.Nrng)
    N_posterior = zeros(length(Nrng))
    for j in 1:length(Nrng)
        N_posterior[j] = count(i->(i==j),Nind)
    end
    results.entropies.entropy_N[i] = entropy(N_posterior/sum(N_posterior))  

    pind = Array(sim.fstate.model.pind)
    prng = Array(sim.fstate.model.prng)
    p_posterior = zeros(length(prng))
    for j in 1:length(prng)
        p_posterior[j] = count(i->(i==j),pind)
    end
    results.entropies.entropy_p[i] = entropy(p_posterior/sum(p_posterior))  

    qind = Array(sim.fstate.model.qind)
    qrng = Array(sim.fstate.model.qrng)
    q_posterior = zeros(length(qrng))
    for j in 1:length(qrng)
        q_posterior[j] = count(i->(i==j),qind)
    end
    results.entropies.entropy_q[i] = entropy(q_posterior/sum(q_posterior))  

    σind = Array(sim.fstate.model.σind)
    σrng = Array(sim.fstate.model.σrng)
    σ_posterior = zeros(length(σrng))
    for j in 1:length(σrng)
        σ_posterior[j] = count(i->(i==j),σind)
    end
    results.entropies.entropy_σ[i] = entropy(σ_posterior/sum(σ_posterior))  
  
    τind = Array(sim.fstate.model.τind)
    τrng = Array(sim.fstate.model.τrng)
    τ_posterior = zeros(length(τrng))
    for j in 1:length(τrng)
        τ_posterior[j] = count(i->(i==j),τind)
    end
    results.entropies.entropy_τ[i] = entropy(τ_posterior/sum(τ_posterior))    

    map = MAP(sim.fstate.model)
    results.map.map_N[i] = map.N
    results.map.map_p[i] = map.p
    results.map.map_q[i] = map.q
    results.map.map_σ[i] = map.σ
    results.map.map_τ[i] = map.τ

end

MAP(sim::NestedFilterSimulation; kwargs...) = MAP(sim.fstate.model; kwargs...)
