function emit(state, model, timestep) end

function emit(
    state::BinomialState{<:Array}, 
    model::AbstractBinomialModel,
    δ::Number
)
    if maximum(size(state.n)) > 1
        error("Emission not supported for non-scalar states")
        return nothing
    else
        q = model.q[1]
        σ = model.σ[1]
        k = state.k[1,1]
        EPSP = Float32(rand(Normal(q*k, σ)))
        return BinomialObservation(EPSP, δ)
    end
end
