function propagate_emit!(state::BinomialState{<:Array}, model::AbstractBinomialModel; dt = nothing, λ = 0.121)
    if maximum(size(state.n)) > 1
        error("Emission not supported for non-scalar states")
        return nothing
    else
        if dt == nothing
            delta = rand(Exponential(λ))
        else
            delta = dt
        end
        propagate!(state, model, delta)
        q = model.q[1]
        sigma = model.sigma[1]
        k = state.k[1,1]
        EPSP = rand(Normal(q*k, sigma))
        return BinomialObservation(EPSP, delta)
    end
end
