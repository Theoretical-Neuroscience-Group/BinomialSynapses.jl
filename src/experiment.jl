struct NestedFilterExperiment{T1, T2, T3}
    filter::T1
    fstate::T2
    tsteps::T3
end

function NestedFilterExperiment(
    Nrng, prng, qrng, σrng, τrng,
    m_out, m_in, width;
    timestep
)
    filter = NestedParticleFilter(width)
    fstate = NestedParticleState(
                m_out, m_in,
                Nrng, prng, qrng, σrng, τrng
             )
    return NestedFilterExperiment(filter, fstate, timestep)
end


function propagate!(sim::NestedFilterExperiment, epsc, dt)
    obs = BinomialObservation(epsc, dt)
    filter_update!(sim, obs)
end

function filter_update!(sim::NestedFilterExperiment, obs)
    return update!(sim.fstate, obs, sim.filter)
end

function run_experiment!(
    sim::NestedFilterExperiment,
    epscs, dts; 
    T::Integer, 
    plot_each_timestep::Bool = false 
    # recording::Recording = NoRecording
)
    for i in 1:T
        propagate!(sim, epscs[i], dts[i])
        if plot_each_timestep
            posterior_plot(sim,i)
        end
    end
			
    map = MAP(sim.fstate.model)
    N_star = map.N
    p_star = map.p
    q_star = map.q
    σ_star = map.σ
    τ_star = map.τ
			
    entrop = zeros(length(keys(sim.tsteps.train)))
    for j in 1:length(keys(sim.tsteps.train))
	train = sim.tsteps.train[j]
	T1 = ScalarBinomialModel(N_star, p_star, q_star, σ_star, τ_star)
    	T2 = sim.filter
    	T3 = ScalarBinomialState(N_star, 0)
    	T4 = deepcopy(sim.fstate)
    	T5 = sim.tsteps
    	T6 = epscs
    	T7 = dts
	entropy_temp = []
        for l in 1:2

            sim_copy = NestedFilterSimulation(T1,T2,T3,T4,T5,T6,T7)
            for k in 1:length(train)
                propagate!(sim_copy,train[k])
            end
            append!(entropy_temp,compute_entropy(sim_copy.fstate.model))
        end
        entrop[j] = mean(entropy_temp)
    end
    print(sim.tsteps.train[argmin(entrop)])
    print('\n')
end

function compute_entropy(model)
    Nind = Array(model.Nind)
    pind = Array(model.pind)
    qind = Array(model.qind)
    σind = Array(model.σind)
    τind = Array(model.τind)

    Nrng = Array(model.Nrng)
    prng = Array(model.prng)
    qrng = Array(model.qrng)
    σrng = Array(model.σrng)
    τrng = Array(model.τrng)

    samples = [Nrng[Nind]';prng[pind]';qrng[qind]';σrng[σind]';τrng[τind]']
   # Σ_est = cov(samples')
    method = LinearShrinkage(DiagonalUnequalVariance(), 0.5)
    Σ_est = cov(method, samples')
			
			
    determinant = det(2*pi*ℯ*Σ_est)
    ent = 0.5*log(determinant)

    return ent

end
