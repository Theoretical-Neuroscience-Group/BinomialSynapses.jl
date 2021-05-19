struct NestedFilterSimulation{T1, T2, T3, T4}
    hmodel::T1
    filter::T2
    hstate::T3
    fstate::T4
end

struct Results{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11}
    entropies::T1
    runtime::T2
    dt::T3    
    e::T4
    n::T5 
    x::T6 
    N_MAP::T7 
    p_MAP::T8 
    q_MAP::T9 
    sigma_MAP::T10 
    tau_MAP::T11 
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
    filterstate = update!(sim.fstate, obs, sim.filter)
    return obs
end

function save_results!(results::Results, sim::NestedFilterSimulation, obs::BinomialObservation, runtime, i, x)
   
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
    results.e[i] = obs.EPSP
    results.n[i] = sim.hstate.n[1,1]
    results.x[i] = x
    map = MAP(sim.fstate.model)
    results.N_MAP[i] = map[:N]
    results.p_MAP[i] = map[:p]
    results.q_MAP[i] = map[:q]
    results.sigma_MAP[i] = map[:σ]
    results.tau_MAP[i] = map[:τ]
end

function run!(sim::NestedFilterSimulation; T::Int, plot_each_timestep = false, protocol = "exponential", parameter = 0.121, record_results = false, penalty = 0)
    times = zeros(0)
    epsps = zeros(0)
    time = 0.
    delta = 0.
    results = Results(zeros(T),zeros(T),zeros(T),zeros(T),zeros(T),zeros(T),zeros(T),zeros(T),zeros(T),zeros(T),zeros(T))
    
    for i in 1:T

        if protocol == "OED"
            runtime = @elapsed obs = propagate!(sim, dt = delta)
        elseif protocol == "OED_exact"
            runtime = @elapsed obs = propagate!(sim, dt = delta)
        elseif protocol == "OED_penalty"
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
        
        x = 1
        if i < T && protocol == "OED"
            runtime2 = @elapsed delta, x = OED(sim, parameter, times, i)
            runtime = runtime + runtime2
        elseif i < T && protocol == "OED_exact"
            runtime2 = @elapsed delta, x = OED_exact(sim, parameter, times, i)
            runtime = runtime + runtime2
         elseif i < T && protocol == "OED_penalty"
            runtime2 = @elapsed delta, x = OED_penalty(sim, parameter, times, i, penalty)
            runtime = runtime + runtime2
        end

        if plot_each_timestep
            posterior_plot(sim.fstate, times, epsps, truemodel = sim.hmodel)
        end
        
        if record_results
            save_results!(results, sim, obs, runtime, i,x)
            if i == T
                if protocol == "uniform"
                    save(string(Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"]),string(parameter[end]),".jld"), "entropies", results.entropies, "dt", results.dt)
                else
                    save(string(Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"]),string(parameter),".jld"), "entropies", results.entropies, "dt", results.dt)
                end                                         
            end
        end
    end
    return times, epsps
end

function OED(sim::NestedFilterSimulation, deltat_candidates, times, i)

    map = MAP(sim.fstate.model)
    N_star = map[:N]
    p_star = map[:p]
    q_star = map[:q]
    sigma_star = map[:σ]
    tau_star = map[:τ]
   
    x = 1
    x = 1-(1-(1-p_star)*x)
    if i>1
        for ii in 2:i
            x = 1-(1-(1-p_star)*x)*exp(-(times[ii]-times[ii-1])/tau_star)

        end
    end
   
    e_temp = zeros(length(deltat_candidates))
    for kk in 1:length(e_temp)
        x_temp = 1-(1-(1-p_star)*x)*exp(-deltat_candidates[kk]/tau_star)
        e_temp[kk] = x_temp*N_star*p_star*q_star
    end

    #sim_local = deepcopy(sim)
    #state = sim_local.fstate.state
    #model = sim_local.fstate.model
    #jitter!(model, sim_local.filter.jittering_width)
    #idx_candidates = sample(1:5,10)
    #candidates = CuVector(deltat_candidates[idx_candidates])
    #e = CuVector(e_temp[idx_candidates])
    #propagate!(state, model, candidates)
   
    h = zeros(length(e_temp))
    for kk in 1:length(e_temp)
        sim_local = deepcopy(sim)
        obs = BinomialObservation(e_temp[kk], deltat_candidates[kk])
        update!(sim_local.fstate, obs, sim_local.filter)
        τind = Array(sim_local.fstate.model.τind)
        τrng = Array(sim_local.fstate.model.τrng)
        τ_posterior = zeros(length(τrng))
        for j in 1:length(τrng)
            τ_posterior[j] = count(i->(i==j),τind)
        end
        h[kk] = entropy(τ_posterior/sum(τ_posterior))
    end    
    return deltat_candidates[argmin(h)], x
        
end

function OED_exact(sim::NestedFilterSimulation, deltat_candidates, times, i)

    N_star = sim.hmodel.N[1]
    p_star = sim.hmodel.p[1]
    q_star = sim.hmodel.q[1]
    sigma_star = sim.hmodel.σ[1]
    tau_star = sim.hmodel.τ[1]

   
    x = 1
    x = 1-(1-(1-p_star)*x)
    if i>1
        for ii in 2:i
            x = 1-(1-(1-p_star)*x)*exp(-(times[ii]-times[ii-1])/tau_star)
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
        τind = Array(sim_local.fstate.model.τind)
        τrng = Array(sim_local.fstate.model.τrng)
        τ_posterior = zeros(length(τrng))
        for j in 1:length(τrng)
            τ_posterior[j] = count(i->(i==j),τind)
        end
        h[kk] = entropy(τ_posterior/sum(τ_posterior))
    end    

    return deltat_candidates[argmin(h)], x
        
end

function OED_penalty(sim::NestedFilterSimulation, deltat_candidates, times, i, penalty)

    map = MAP(sim.fstate.model)
    N_star = map[:N]
    p_star = map[:p]
    q_star = map[:q]
    sigma_star = map[:σ]
    tau_star = map[:τ]
   
    x = 1
    x = 1-(1-(1-p_star)*x)
    if i>1
        for ii in 2:i
            x = 1-(1-(1-p_star)*x)*exp(-(times[ii]-times[ii-1])/tau_star)

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
        τind = Array(sim_local.fstate.model.τind)
        τrng = Array(sim_local.fstate.model.τrng)
        τ_posterior = zeros(length(τrng))
        for j in 1:length(τrng)
            τ_posterior[j] = count(i->(i==j),τind)
        end
        h[kk] = entropy(τ_posterior/sum(τ_posterior)) + penalty*deltat_candidates[kk]
    end    
    return deltat_candidates[argmin(h)], x
        
end



MAP(sim::NestedFilterSimulation) = MAP(sim.fstate.model)
variance(sim::NestedFilterSimulation) = variance(sim.fstate.model)
ent(sim::NestedFilterSimulation) = ent(sim.fstate.model)
