relu(x) = max(x, 0)

function propagate!(n::AbstractArray, k::AbstractArray, model::AbstractBinomialModel, dt)
    # propagate the hidden states n, k by one timestep

    # refill counts and probabilities
    @. n     = relu(model.N - n + k)
    p_refill = @. 1 - exp(-dt / model.τ)

    # draw vesicles to be refilled, update n
    _sample_binomial!(k, n, p_refill)
    @. n     = model.N - n + k

    # draw vesicles to be released
    _sample_binomial!(k, n, model.p)
    return n, k
end

function _sample_binomial!(k::AbstractArray, n, p)
    @. k     = rand(Binomial(n, p))
    return k
end

function _sample_binomial!(k::AnyCuArray, n, p)
    BinomialGPU.rand_binomial!(k, count = n, prob = p)
    return k
end

"""
    propagate!(state, model, dt)

Propagate `state` forward according to the `model`, given a time step `dt`.
"""
function propagate!(state::BinomialState, model, dt)
    propagate!(state.n, state.k, model, dt)
    return state
end
