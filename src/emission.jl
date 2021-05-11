function propagate_emit!(
    state, 
    model, 
    timestep::Timestep = RandomTimestep(Exponential(0.121))
) 
    δ = get_step(timestep)
    propagate_emit!(state, model, δ)
end

function propagate_emit!(
    state::BinomialState{<:Array}, 
    model::AbstractBinomialModel,
    δ::Number
)
    if maximum(size(state.n)) > 1
        error("Emission not supported for non-scalar states")
        return nothing
    else
        propagate!(state, model, δ)
        q = model.q[1]
        σ = model.σ[1]
        k = state.k[1,1]
        EPSP = rand(Normal(q*k, σ))
        return BinomialObservation(EPSP, δ)
    end
end
