struct NestedFilterSimulation{T1, T2, T3, T4}
    hmodel::T1
    filter::T2
    hstate::T3
    fstate::T4
end

struct Results{T1, T2, T3}
    entropies::T1
    runtime::T2
    dt::T3    
end

function NestedFilterSimulation(
    N, p, q, σ, τ,
    Nrng, prng, qrng, σrng, τrng,
    m_out, m_in, width;
    λ = 0.121,
    dt = nothing
)
    hmodel = ScalarBinomialModel(N, p, q, σ, τ)
    filter = NestedParticleFilter(width)
    hstate = ScalarBinomialState(N, 0)
    fstate = NestedParticleState(
                m_out, m_in,
                Nrng, prng, qrng, σrng, τrng
             )
    return NestedFilterSimulation(hmodel, filter, hstate, fstate)
end

function propagate!(sim::NestedFilterSimulation; dt = nothing, λ = nothing)
    obs = propagate_emit!(sim.hstate, sim.hmodel, dt = dt, λ = λ)
    update!(sim.fstate, obs, sim.filter)
    return obs
end

function save_results!(results::Results, sim::NestedFilterSimulation, obs::BinomialObservation, runtime, i)
   
    Nind = Array(sim.fstate.model.Nind)
    pind = Array(sim.fstate.model.pind)
    qind = Array(sim.fstate.model.qind)
    σind = Array(sim.fstate.model.σind)
    τind = Array(sim.fstate.model.τind)
    Nrng = Array(sim.fstate.model.Nrng)
    prng = Array(sim.fstate.model.prng)
    qrng = Array(sim.fstate.model.qrng)
    σrng = Array(sim.fstate.model.σrng)
    τrng = Array(sim.fstate.model.τrng)
    
    N_posterior = zeros(length(Nrng))
    p_posterior = zeros(length(prng))
    q_posterior = zeros(length(qrng))
    σ_posterior = zeros(length(σrng))
    τ_posterior = zeros(length(τrng))

    for j in 1:length(Nrng)
        N_posterior[j] = count(i->(i==j),Nind)
    end
    for j in 1:length(prng)
        p_posterior[j] = count(i->(i==j),pind)
    end
    for j in 1:length(qrng)
        q_posterior[j] = count(i->(i==j),qind)
    end
    for j in 1:length(σrng)
        σ_posterior[j] = count(i->(i==j),σind)
    end
    for j in 1:length(τrng)
        τ_posterior[j] = count(i->(i==j),τind)
    end

    results.entropies[i] = entropy(τ_posterior/sum(τ_posterior))
    results.runtime[i] = runtime
    results.dt[i] = obs.dt
end

function run!(sim::NestedFilterSimulation; T::Int, plot_each_timestep = false, protocol = "exponential", parameter = 0.121, record_results = false)
    times = zeros(0)
    epsps = zeros(0)
    time = 0.
    delta = 0.
    results = Results(zeros(T),zeros(T),zeros(T))
    
    for i in 1:T
        print(string("delta=",delta))
        print("\n")
        if protocol == "OED"
            runtime = @elapsed obs = propagate!(sim, dt = delta)
        elseif protocol == "exponential"
            runtime = @elapsed obs = propagate!(sim, λ = parameter)
        elseif protocol == "constant"
            runtime = @elapsed obs = propagate!(sim, dt = parameter)
        elseif protocol == "uniform"
            runtime = @elapsed obs = propagate!(sim, dt = rand(parameter))
        end   
        
        push!(times, time += obs.dt)
        push!(epsps, obs.EPSP)
        
        if i < T && protocol == "OED"
            runtime2 = @elapsed delta = OED(sim, parameter, times, i)
            runtime = runtime + runtime2
        end

        if plot_each_timestep
            posterior_plot(sim.fstate, times, epsps, truemodel = sim.hmodel)
        end
        
        if record_results
            save_results!(results, sim, obs, runtime, i)
            if i == T
                save(string(Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"]),".jld"), "entropies", results.entropies, "runtime", results.runtime, "dt", results.dt)
            end
        end

    end
    return times, epsps
end

function OED(sim::NestedFilterSimulation, deltat_candidates, times, i)

    
    map = MAP(sim)
    N_star = map[:N]
    p_star = map[:p]
    q_star = map[:q]
    sigma_star = map[:σ]
    tau_star = map[:τ]
    
    x = 1
    if i>1
        for ii in 2:i
            x = 1-(1-(1-p_star)*x)*exp(-(times[ii]-times[ii-1])/tau_star)
            print(times[ii]-times[ii-1])
            print("\n")
        end
    end
    e_temp = zeros(length(deltat_candidates))
    for kk in 1:length(e_temp)
        x_temp = 1-(1-(1-p_star)*x)*exp(-deltat_candidates[kk]/tau_star)
        e_temp[kk] = x_temp*N_star*p_star*q_star
    end

    
    h = zeros(length(e_temp))
    for kk in 1:length(e_temp)
        sim_local = deepcopy(sim)
        obs = BinomialObservation(e_temp[kk], deltat_candidates[kk])
        update!(sim_local.fstate, obs, sim_local.filter)
        #v = ent(sim_local)
        #h[kk] = v[:τ]
        τind = Array(sim_local.fstate.model.τind)
        τrng = Array(sim_local.fstate.model.τrng)
        τ_posterior = zeros(length(τrng))
        for j in 1:length(τrng)
            τ_posterior[j] = count(i->(i==j),τind)
        end
        h[kk] = entropy(τ_posterior/sum(τ_posterior))
    end

    
    return deltat_candidates[argmin(h)] 
    
end

MAP(sim::NestedFilterSimulation) = MAP(sim.fstate.model)
variance(sim::NestedFilterSimulation) = variance(sim.fstate.model)
ent(sim::NestedFilterSimulation) = ent(sim.fstate.model)
