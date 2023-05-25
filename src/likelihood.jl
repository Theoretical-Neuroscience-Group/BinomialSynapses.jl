"""
    likelihood(state, model::AbstractBinomialModel, obs)
Return the likelihood of an observation conditioned on the current state and model.
This broadcasts properly over state and model ensemble, if they have compatible sizes.
"""
likelihood(state::BinomialState, model, obs) = likelihood(state.k, model, obs)

function likelihood(k, model::AbstractBinomialModel, obs)
    return mean(
                exp.(-0.5f0 .* ((obs .- model.q .* k) ./ model.σ).^2)
                ./ (sqrt(2*Float32(pi)) .* model.σ)
           , dims = 2
           )[:,1]
end

gauss(x, μ, σ) = exp(-((x-μ)/σ)^2/2)

function likelihood_indices(
    k::AnyCuArray,
    model::AbstractBinomialModel, 
    obs,
    rm::ResamplingMethod
)
    v = gauss.(obs, model.q .* k, model.σ)
    u, idx = indices!(v, rm)
    
    # normalization of u
    α = last(size(k)) * sqrt(2*Float32(pi))
    u ./= (α .* model.σ)
    return u, idx
end

"""
    likelihood_resample!(state, model::AbstractBinomialModel, obs)
Return the likelihood of an observation conditioned on the current state and model ensemble and at the same time resample the state ensemble (inner particles).
"""
function likelihood_resample!(
    state::BinomialState,
    model,
    obs::BinomialObservation,
    rm::ResamplingMethod
)
    u, idx = likelihood_indices(state.k, model, obs.EPSP, rm)
    resample!(state, idx)
    return u
end
