struct NestedFilterExperiment{T1, T2, T3}
    filter::T1
    fstate::T2
    tsteps::T3
end

function show_histogram(
    bins, indices1, indices2;
    truevalue = nothing, xlabel = L"N [-]", ylabel = L"p(N)"
)
    nbins = length(bins)
	
    counts1 = zeros(nbins)
    for i in 1:length(indices1)
        counts1[indices1[i]] += 1
    end
    counts2 = zeros(nbins)
    for i in 1:length(indices2)
        counts2[indices2[i]] += 1
    end
	
    p = plot(
            bins,
            counts1./sum(counts1),
            xlabel=xlabel,
            ylabel=ylabel,
            legend=false
           )
    plot!(
            bins,
            counts2./sum(counts2),
            xlabel=xlabel,
            ylabel=ylabel,
            legend=false
           )
	
    if !isnothing(truevalue)
        plot!([truevalue], seriestype="vline",legend=false)
    end
    return p
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
	
function postpro_experiment!(
    sim1::NestedFilterExperiment,
    sim2::NestedFilterExperiment,
    epscs1, dts1,
    epscs2, dts2;
    T::Integer, 
    plot_each_timestep::Bool = false 
    # recording::Recording = NoRecording
)
    for i in 1:T
        propagate!(sim1, epscs1[i], dts1[i])
	propagate!(sim2, epscs2[i], dts2[i])
        if plot_each_timestep
            
	    fstate1 = sim1.fstate        
	    Nrng = Array(fstate1.model.Nrng)
	    prng = Array(fstate1.model.prng)
	    qrng = Array(fstate1.model.qrng)
	    σrng = Array(fstate1.model.σrng)
	    τrng = Array(fstate1.model.τrng)
	    Nind1 = Array(fstate1.model.Nind)
	    pind1 = Array(fstate1.model.pind)
	    qind1 = Array(fstate1.model.qind)
	    σind1 = Array(fstate1.model.σind)
	    τind1 = Array(fstate1.model.τind)
				
	    fstate2 = sim2.fstate
	    Nind2 = Array(fstate2.model.Nind)
	    pind2 = Array(fstate2.model.pind)
	    qind2 = Array(fstate2.model.qind)
	    σind2 = Array(fstate2.model.σind)
	    τind2 = Array(fstate2.model.τind)
				
	    pN = show_histogram(Nrng, Nind1, Nind2,
		    xlabel = L"N [-]", ylabel = L"p(N)")
	    pp = show_histogram(prng, pind1, pind2,
		    xlabel = L"p [-]", ylabel = L"p(p)")
	    pq = show_histogram(qrng, qind1, qind2,
		    xlabel = L"q [A]", ylabel = L"p(q)")
	    pσ = show_histogram(σrng, σind1, σind2,
		    xlabel = L"\sigma [A]", ylabel = L"p(\sigma)")
	    pτ = show_histogram(τrng, τind1, τind2,
		    xlabel = L"\tau [s]", ylabel = L"p(\tau)")

	    display(plot(pN, pN, pp, pq, pσ, pτ, layout = (3, 2)))
	    if i%5 == 0
		savefig(string(i,".png"))
	    end
        end
    end

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

    samples = [Nrng[Nind]'; prng[pind]'; qrng[qind]'; σrng[σind]'; τrng[τind]']
    # Σ_est = cov(samples')
    method = LinearShrinkage(DiagonalUnequalVariance(), 0.5)
    Σ_est = cov(method, samples, dims = 2)
    ent = entropy(MvNormal(Σ_est))
    return ent
end
