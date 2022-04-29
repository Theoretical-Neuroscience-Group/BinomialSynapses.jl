"""
    likelihood(state, model::AbstractBinomialModel, obs)

Return the likelihood of an observation conditioned on the current state and model.
This broadcasts properly over state and model ensemble, if they have compatible sizes.
"""
likelihood(state::BinomialState, model, obs) = likelihood(state.k, model, obs)

function likelihood(k, model::AbstractBinomialModel, obs)
    T = eltype(model.σ)
    return mean(
                exp.(-T(0.5) .* ((obs .- model.q .* k) ./ model.σ).^2)
                ./ (sqrt(2*T(pi)) .* model.σ)
           , dims = 2
           )[:,1]
end

gauss(x, μ, σ) = exp(-((x-μ)/σ)^2/2)

function likelihood_indices(
    k::AnyCuArray,
    model::AbstractBinomialModel, 
    obs
)
    v = gauss.(obs, model.q .* k, model.σ)
    u, idx = indices!(v)
    
    # normalization of u
    T = eltype(model.σ)
    α = last(size(k)) * sqrt(2*T(pi))
    u ./= (α .* model.σ)
    return u, idx
end

"""
    likelihood_resample!(state, model::AbstractBinomialModel, obs)

Return the likelihood of an observation conditioned on the current state and model ensemble and at the same time resample the state ensemble (inner particles).
"""
function likelihood_resample!(state::BinomialState, model, obs::BinomialObservation)
    u, idx = likelihood_indices(state.k, model, obs.EPSP)
    resample!(state, idx)
    return u
end
