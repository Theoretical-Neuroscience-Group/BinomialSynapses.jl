"""
    NestedFilterSimulation(hmodel, filter, hstate, fstate, tsteps, times, epsps)

This object stores static (`hmodel`, `filter`, `tsteps`) and dynamic (`hstate`, `fstate`, `tsteps`, `times`, `epsps`) information about the simulation.
"""
struct NestedFilterSimulation{T1, T2, T3, T4, T5, T6, T7}
    hmodel::T1
    filter::T2
    hstate::T3
    fstate::T4
    tsteps::T5
    times::T6
    epsps::T7
end

"""
    NestedFilterSimulation(
        N, p, q, σ, τ,
        Nrng, prng, qrng, σrng, τrng,
        m_out, m_in, width;
        timestep::Timestep = RandomTimestep(Exponential(0.121))
    )

This is the main way simulations are supposed to be constructed by the user, i.e.
by specifying 
- ground truth parameters `N`, `p`, `q`, `σ`, `τ`.
- parameter grids for the model ensemble (outer particles) `Nrng`, `prng`,...
- size of the ensemble, i.e. number of outer_particles `m_out` and inner particles `m_in`
- the width of the jittering kernel `width`
- a method for choosing time steps `timestep`
"""
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

"""
    m_out(sim)

Return the number of outer particles of the simulation `sim`.
"""
function m_out(sim::NestedFilterSimulation)
    return size(sim.fstate.state.n, 1)
end

"""
    propagate_hidden!(sim, dt)

Propagate the hidden state for a time step `dt`.
"""
function propagate_hidden!(sim, dt)
    return propagate!(sim.hstate, sim.hmodel, dt)
end

"""
    emit(sim, dt)

Emit an EPSP after time step `dt`.
"""
function emit(sim::NestedFilterSimulation, dt)
    return emit(sim.hstate, sim.hmodel, dt)
end

"""
    filter_update!(sim, obs)

Update the filter state based on observation `obs`.
"""
function filter_update!(sim::NestedFilterSimulation, obs)
    return update!(sim.fstate, obs, sim.filter)
end

"""
    initialize!(sim)

Initialize the simulation.
"""
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
function get_step(sim::NestedFilterSimulation{T1, T2, T3, T4, T5, T6, T7}) where 
{T1, T2, T3, T4, T5 <: DeterministicTrain, T6, T7}
    i = length(sim.times)
    return sim.tsteps.train[i]
end

"""
    propagate!(sim)

Propagate the simulation, i.e. choose a time step and then propagate the simulation by it.
"""
function propagate!(sim::NestedFilterSimulation)
    time1 = @timed get_step(sim)
    dt = time1.value
    time2 = propagate!(sim, dt)
    return time1.time + time2
end

"""
    propagate!(sim, dt)

Propagate the simulation by time step `dt`.
"""
function propagate!(sim::NestedFilterSimulation, dt)
    propagate_hidden!(sim, dt)
    obs = emit(sim, dt)
    time2 = @timed filter_update!(sim, obs)
    push!(sim.times, sim.times[end] + dt)
    push!(sim.epsps, obs.EPSP)
    return time2.time
end

"""
    run!(
        sim; 
        T, 
        plot_each_timestep = false, 
        recording = NoRecording
    )

Run a simulation for `T` time steps.
Set `plot_each_timestep = true` to get a live update of the simulation (this will reduce performance) and set `recording` to collect data while running the simulation (see `Recording`).
"""
function run!(
    sim::NestedFilterSimulation; 
    T::Integer, 
    plot_each_timestep::Bool = false, 
    recording::Recording = NoRecording
)
    if length(sim.times) == 0
        initialize!(sim)
    end
    for i in 1:T
        begin
            time = propagate!(sim)
        end
        if plot_each_timestep
            posterior_plot(sim,i)
        end
        update!(recording, sim, time) 
    end
    save(recording)
    return sim.times, sim.epsps
end

function run_adaptative_eta!(
    sim::NestedFilterSimulation; 
    T::Integer, 
    plot_each_timestep::Bool = false, 
    recording::Recording = NoRecording,
    α = 0
)
    if length(sim.times) == 0
        initialize!(sim)
    end
    for i in 1:T
        begin
            print(i)
            print("\n")
            print(sim.tsteps.penalty)
            print("\n")
            prev_ent = compute_entropy(sim.fstate.model)
            print(prev_ent)
            print("\n")
            time = propagate!(sim)
            new_ent = compute_entropy(sim.fstate.model)
            print(new_ent)
            print("\n")            
            η = deepcopy(sim.tsteps.penalty)
            Δ_ent = new_ent - prev_ent
            sim.tsteps.penalty = α*Δ_ent/sim.times[i] - α*η
            print(sim.tsteps.penalty)
            print("\n")
        end
        if plot_each_timestep
            posterior_plot(sim,i)
        end
        update!(recording, sim, time) 
    end
    save(recording)
    return sim.times, sim.epsps
end

function estimate_posterior!(
    sim::NestedFilterExperiment,
    epscs, dts; 
    T::Integer, 
    plot_each_timestep::Bool = false, 
    recording::Recording = NoRecording
)

    for i in 1:T
        begin
    	    obs = BinomialObservation(epscs[i], dts[i])
            time2 = @timed filter_update!(sim, obs)
        end
        if plot_each_timestep
            posterior_plot(sim,i)
        end
        update!(recording, sim, time2.time) 
    end
    save(recording)
end



function runBatch_map!(
    sim::NestedFilterSimulation;
    T::Integer,
    plot_each_timestep::Bool = false,
    recording::Recording = NoRecording
)
    if length(sim.times) == 0
        initialize!(sim)
    end
    for i in 1:T
        time_batch = @timed begin
            entrop = zeros(length(keys(sim.tsteps.train)))
            for j in 1:length(keys(sim.tsteps.train))
                train = sim.tsteps.train[j]

                entropy_temp = []
				
		map = MAP(sim.fstate.model)

    		N_star = map.N
    		p_star = map.p
    		q_star = map.q
		σ_star = map.σ
    		τ_star = map.τ
				
		T1 = ScalarBinomialModel(N_star, p_star, q_star, σ_star, τ_star)
    		T2 = sim.filter
    		T3 = ScalarBinomialState(N_star, 0)
    		T4 = deepcopy(sim.fstate)
    		T5 = sim.tsteps
    		T6 = deepcopy(sim.times)
    		T7 = deepcopy(sim.epsps)
                for l in 1:5

                    sim_copy = NestedFilterSimulation(T1,T2,T3,T4,T5,T6,T7)
                    for k in 1:length(train)
                        propagate!(sim_copy,train[k])
                    end
                    append!(entropy_temp,compute_entropy(sim_copy.fstate.model))
                end
                entrop[j] = mean(entropy_temp)
            end
	end
	print(time_batch.time)
	print("\n")
        train_opt = sim.tsteps.train[argmin(entrop)]
	for j in 1:length(train_opt)
	    begin
		time = propagate!(sim,train_opt[j])
	    end
	    if plot_each_timestep
		posterior_plot(sim,j)
	    end
	    update!(recording, sim, time)
	end
    end
    save(recording)
    return sim.times, sim.epsps
end	
	
"""
Instead of using point-based estimates for theta and the next observation, here the expected next entropy is evaluated using MC samples.
"""
	
function run_exact_1!(
    sim::NestedFilterSimulation;
    T::Integer,
    plot_each_timestep::Bool = false,
    recording::Recording = NoRecording,
    M = 100
)
    if length(sim.times) == 0
        initialize!(sim)
    end
    for i in 1:T
        begin
            entrop = zeros(length(sim.tsteps.dts))
            for j in 1:length(entrop)
                dt = sim.tsteps.dts[j]
                entropy_temp = []
	            for k in 1:M
                    sim_copy = deepcopy(sim)
                    m_out = length(sim_copy.fstate.model.N)
                    random_idx = rand(1:m_out)
                    N_star = Array(sim_copy.fstate.model.N)[random_idx]
                    p_star = Array(sim_copy.fstate.model.p)[random_idx]
                    q_star = Array(sim_copy.fstate.model.q)[random_idx]
                    σ_star = Array(sim_copy.fstate.model.σ)[random_idx]
                    τ_star = Array(sim_copy.fstate.model.τ)[random_idx]
                    x = 1.
                    times = sim.times
                    L = length(times)
                    if L > 1
                        for ii in 2:L
                            x = 1-(1-(1-p_star)*x)*exp(-(times[ii]-times[ii-1])/τ_star)
                        end
                    end
                    x_temp = 1-(1-(1-p_star)*x)*exp(-dt/τ_star)
                    e_temp = x_temp*N_star*p_star*q_star
                    obs = BinomialObservation(e_temp, dt)
                    update!(sim_copy.fstate, obs, sim_copy.filter)
                    append!(entropy_temp,compute_entropy(sim_copy.fstate.model))
		        end
		        entrop[j] = mean(entropy_temp)
	        end
            dt_opt = sim.tsteps.dts[argmin(entrop)]
            time = propagate!(sim,dt_opt)
        end
        if plot_each_timestep
            posterior_plot(sim,j)
        end
	    update!(recording, sim, time)
    end
    save(recording)
    return sim.times, sim.epsps
end
	
function run_exact_2!(
    sim::NestedFilterSimulation;
    T::Integer,
    plot_each_timestep::Bool = false,
    recording::Recording = NoRecording,
    M = 100
)
    if length(sim.times) == 0
        initialize!(sim)
    end
    for i in 1:T
        begin
            entrop = zeros(length(sim.tsteps.dts))
            for j in 1:length(entrop)
                dt = sim.tsteps.dts[j]
                entropy_temp = []
	            for k in 1:M
                    sim_copy = deepcopy(sim)
                    m_out, m_in = size(Array(sim_copy.fstate.state.n))
                    random_idx_out = rand(1:m_out)
                    random_idx_in = rand(1:m_in)

                    N_star = Array(sim_copy.fstate.model.N)[random_idx_out]
                    p_star = Array(sim_copy.fstate.model.p)[random_idx_out]
                    q_star = Array(sim_copy.fstate.model.q)[random_idx_out]
                    σ_star = Array(sim_copy.fstate.model.σ)[random_idx_out]
                    τ_star = Array(sim_copy.fstate.model.τ)[random_idx_out]

                    n_ = Array(sim_copy.fstate.state.n)[random_idx_out,random_idx_in]
                    k_ = Array(sim_copy.fstate.state.k)[random_idx_out,random_idx_in]
                    
                    p_refill = 1 - exp(-dt / τ_star)
                    n_  = relu(N_star - n_ + k_)
                    k_ = rand(Binomial(n_, p_refill))
                    n_ = N_star - n_ + k_
                    k_ = rand(Binomial(n_, Float64(p_star)))
                    e_temp = Float32(rand(Normal(q_star*k_, σ_star)))

                    obs = BinomialObservation(e_temp, dt)
                    update!(sim_copy.fstate, obs, sim_copy.filter)
                    append!(entropy_temp,compute_entropy(sim_copy.fstate.model))
		        end
		        entrop[j] = mean(entropy_temp)
	        end
            dt_opt = sim.tsteps.dts[argmin(entrop)]
            time = propagate!(sim,dt_opt)
        end
        if plot_each_timestep
            posterior_plot(sim,j)
        end
	    update!(recording, sim, time)
    end
    save(recording)
    return sim.times, sim.epsps
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

    samples = [Nrng[Nind]'./Nrng[end];prng[pind]'./prng[end];qrng[qind]'./qrng[end];σrng[σind]'./σrng[end];τrng[τind]'./τrng[end]]
    Σ_est = cov(samples')			
			
    determinant = det(2*pi*ℯ*Σ_est)
    if determinant > 0
        ent = 0.5*log(determinant)
    else
        ent = Inf
    end
    return ent

end



MAP(sim::NestedFilterSimulation; kwargs...) = MAP(sim.fstate.model; kwargs...)

"""
    Recording(f1, f2, sim)

Initialize a `Recording` for the specified functions `f1` and `f2` and already existing simulation `sim`.
"""
function Recording(f1, f2, sim::NestedFilterSimulation)
    begin
        time = @timed nothing
    end
    res = f1(sim, time.time)
    data = [res]
    return Recording(f1, f2, data)
end
		

		
	



