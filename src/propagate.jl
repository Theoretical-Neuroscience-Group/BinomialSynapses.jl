relu(x) = max(x, 0)

function propagate!(n::AbstractArray, k::AbstractArray, model::AbstractBinomialModel, dt)
    # propagate the hidden states n, k by one timestep
    m_out, m_in = size(n)

    @inbounds for i in 1:m_in
        @inbounds for j in 1:m_in
            # refill counts and probabilities
            n[i, j]  = relu(model.N[i] - n[i, j] + k[i, j])
            p_refill = 1 - exp(-dt / model.τ[i])

            # draw vesicles to be refilled, update n
            k[i, j] = rand(Binomial(n[i, j], p_refill))
            n[i, j] = model.N[i] - n[i, j] + k[i, j]

            # draw vesicles to be released
            k[i, j] = rand(Binomial(n[i, j], model.p[i]))
        end
    end
    return n, k
end

function propagate!(n::AnyCuArray, k::AnyCuArray, model::AbstractBinomialModel, dt)
    # propagate the hidden states n, k by one timestep

    # refill counts and probabilities
    @. n     = relu(model.N - n + k)
    p_refill = @. 1 - exp(-dt / model.τ)

    # draw vesicles to be refilled, update n
    BinomialGPU.rand_binomial!(k, count = n, prob = p_refill)
    @. n     = model.N - n + k

    # draw vesicles to be released
    BinomialGPU.rand_binomial!(k, count = n, prob = model.p)
    return n, k
end

function propagate!(state::BinomialState, model, dt)
    propagate!(state.n, state.k, model, dt)
    return state
end
