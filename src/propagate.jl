relu(x) = max(x, 0)

function propagate!(n, k, model::BinomialModel, dt)
    # propagate the hidden states n, k by one timestep

    # refill counts and probabilities
    @. n     = relu(model.N - n + k)
    p_refill = @. 1 - exp(-dt / model.tau)

    # draw vesicles to be refilled, update n
    BinomialGPU.rand_binomial!(k, count = n, prob = p_refill)
    @. n     = model.N - n + k

    # draw vesicles to be released
    BinomialGPU.rand_binomial!(k, count = n, prob = model.p)
    return n, k
end
