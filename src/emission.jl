function propagate_emit!(state::BinomialState{<:Array}, model::AbstractBinomialModel; dt = nothing, λ = 0.121)
    if maximum(size(state.n)) > 1
        error("Emission not supported for non-scalar states")
        return nothing
    else
        if dt == nothing
            δ = rand(Exponential(λ))
        else
            δ = dt
        end
        propagate!(state, model, δ)
        q = model.q[1]
        σ = model.σ[1]
        k = state.k[1,1]
        EPSP = rand(Normal(q*k, σ))
        return BinomialObservation(EPSP, δ)
    end
end
