struct NestedFilterSimulation{T1, T2, T3, T4}
    hmodel::T1
    filter::T2
    hstate::T3
    fstate::T4
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

function save_results(sim::NestedFilterSimulation, obs::BinomialObservation, runtime, i, simulation_number)
   
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

    N_entropy = entropy(N_posterior/sum(N_posterior))
    p_entropy = entropy(p_posterior/sum(p_posterior))
    q_entropy = entropy(q_posterior/sum(q_posterior))
    σ_entropy = entropy(σ_posterior/sum(σ_posterior))
    τ_entropy = entropy(τ_posterior/sum(τ_posterior))
    
    if isdir(string(simulation_number))==false
        mkdir(string(simulation_number))
    end
    
    cd(string(simulation_number))
    save(string(i,".jld"), "e", obs.EPSP,
        "dt", obs.dt,
        "N_entropy", N_entropy,
        "p_entropy", p_entropy,
        "q_entropy", q_entropy,
        "sigma_entropy", σ_entropy,
        "tau_entropy", τ_entropy,
        "runtime",runtime)
    cd("..")
end

function run!(sim::NestedFilterSimulation; T::Int, plot_each_timestep = false, protocol = "exponential", parameter = 0.121, record_results = false)
    times = zeros(0)
    epsps = zeros(0)
    time = 0.
    delta = 0.
    
    for i in 1:T
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
            save_results(sim, obs, runtime, i, Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"]))
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
    for ii in 1:i
        x = 1-(1-(1-p_star)*x)*exp(-times[ii]/tau_star)
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
        v = variance(sim_local)
        h[kk] = v[:τ]
    end
    
    return deltat_candidates[argmin(h)] 
    
end

MAP(sim::NestedFilterSimulation) = MAP(sim.fstate.model)
variance(sim::NestedFilterSimulation) = variance(sim.fstate.model)
