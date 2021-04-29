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

function propagate!(sim::NestedFilterSimulation)
    obs = propagate_emit!(sim.hstate, sim.hmodel)
    update!(sim.fstate, obs, sim.filter)
    return obs
end

function propagate_OED!(sim::NestedFilterSimulation, dt)
    obs = propagate_emit!(sim.hstate, sim.hmodel, dt=dt)
    update!(sim.fstate, obs, sim.filter)
    return obs
end

function run!(sim::NestedFilterSimulation; T::Int, plot_each_timestep = false)
    times = zeros(0)
    epsps = zeros(0)
    time = 0.

    for i in 1:T
        @time begin
            obs = propagate!(sim)
        end

        push!(times, time += obs.dt)
        push!(epsps, obs.EPSP)
        if plot_each_timestep
            posterior_plot(sim.fstate, times, epsps, truemodel = sim.hmodel)
        end
    end
    return times, epsps
end

function run_OED!(sim::NestedFilterSimulation; T::Int, plot_each_timestep = false)
    times = zeros(0)
    epsps = zeros(0)
    time = 0.
    delta = 0.
    
    for i in 1:T
        
        obs = propagate_OED!(sim,delta)

        push!(times, time += obs.dt)
        push!(epsps, obs.EPSP)        
        
        if i < T
                
            map = MAP(sim)
            N_star = map[:N]
            p_star = map[:p]
            q_star = map[:q]
            sigma_star = map[:σ]
            tau_star = map[:τ]
#
 #           delta_candidates = LinRange(0.05,1,25)
#
 #           x = 1
        #    for ii in 1:i
       #         x = 1-(1-(1-p_star)*x)*exp(-times[ii]/tau_star)
      #      end
     #       e_temp = zeros(25)
    #        for kk in 1:25
   #             x_temp = 1-(1-(1-p_star)*x)*exp(-delta_candidates[kk]/tau_star)
  #              e_temp[kk] = x_temp*N_star*p_star*q_star
 #           end
#
       #     h = zeros(25)
      #      for kk in 1:25
     #           sim_local = sim
    #            obs = BinomialObservation(e_temp[kk], delta_candidates[kk])
   #             update!(sim_local.fstate, obs, sim_local.filter)
  #              v = variance(sim_local)
 #               h[kk] = v[:τ]
#            end
            #delta = delta_candidates[argmin(h)]   
            delta = rand(Exponential(0.121))

        end       
    end
    print(N_star)
    print("\n")
    print(p_star)
    print("\n")
    print(q_star)
    print("\n")
    print(sigma_star)
    print("\n")
    print(tau_star)
    print("\n")
    return times, epsps
end

MAP(sim::NestedFilterSimulation) = MAP(sim.fstate.model)
variance(sim::NestedFilterSimulation) = variance(sim.fstate.model)
