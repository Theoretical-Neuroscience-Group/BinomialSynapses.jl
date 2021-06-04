function likelihood(k, model::AbstractBinomialModel, obs)
    return mean(
                exp.(-0.5f0 .* ((obs .- model.q .* k) ./ model.σ).^2)
                ./ (sqrt(2*Float32(pi)) .* model.σ)
           , dims = 2
           )[:,1]
end

function likelihood_indices(k, model, obs::Number)
    obs_array = CUDA.fill(Float32(obs), size(k)[1:end-1]...)
    return likelihood_indices(k, model, obs_array)
end

gauss(x, μ, σ) = exp(-((x-μ)/σ)^2/2)

function likelihood_indices(
    k::AnyCuArray,
    model::AbstractBinomialModel, 
    obs::AnyCuArray
)
    v = gauss.(obs, model.q .* k, model.σ)
    u, idx = indices!(v)
    
    # normalization of u
    u ./= (last(size(k)) .* sqrt(2*Float32(pi)) .* model.σ)
    return u, idx
end

function likelihood_resample!(state::BinomialState, model, observation::BinomialObservation)
    u, idx = likelihood_indices(state.k, model, observation.EPSP)
    resample!(state, idx)
    return u
end
