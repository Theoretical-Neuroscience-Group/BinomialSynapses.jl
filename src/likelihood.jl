function likelihood(k, model::AbstractBinomialModel, obs)
    return mean(
                exp.(-0.5f0 .* ((obs .- model.q .* k) ./ model.σ).^2)
                ./ (sqrt(2*Float32(pi)) .* model.σ)
           , dims = 2
           )[:,1]
end

function kernel_likelihood_indices!(
    u, v, idx, k, 
    q, σ, obs, 
    r, randstates, 
    Rout, M_in
)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x # physical index
    M_out = length(Rout)
    @inbounds if id <= M_out
        i = Rout[id]
        vsum = 0f0
        CurMax = 1f0
        for j in 1:M_in
            # omitting normalization constant here; it is only needed for u
            vj      = CUDA.exp(-0.5f0 * ((obs[i] - q[i] * k[i, j]) / σ[i])^2)
            vsum   += vj
            v[j, i] = vsum
            # sample descending sequence of sorted random numbers
            # r[M_in,i] >= ... >= r[2,i] >= r[1,i]
            # Algorithm by:
            # Bentley & Saxe, ACM Transactions on Mathematical Software, Vol 6, No 3
            # September 1980, Pages 359--364
            mirrorj = M_in - j + 1 # mirrored index i
            CurMax *= CUDA.exp(CUDA.log(GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates)) / mirrorj)
            r[mirrorj, i] = CurMax
        end
        # compute average likelihood across inner particles
        # (with normalization constant that was omitted from v for speed)
        u[i] = vsum / (M_in * CUDA.sqrt(2*Float32(pi)) * σ[i])
        # O(n) binning algorithm for sorted samples
        bindex = 1 # bin index
        @inbounds for j in 1:M_in
            # scale random numbers (this is equivalent to normalizing v)
            rsample = r[j, i] * vsum
            # checking bindex <= M_in - 1 is redundant since
            # v[M_in, j] = vsum
            while rsample > v[bindex, i]
                bindex += 1
            end
            idx[i, j] = bindex
        end
    end
    return nothing
end

function likelihood_indices(k, model, obs::Number)
    obs_array = CUDA.fill(Float32(obs), size(k)[1:end-1]...)
    return likelihood_indices(k, model, obs_array)
end

function likelihood_indices(
    k::AnyCuArray,
    model::AbstractBinomialModel, 
    obs::AnyCuArray
)
    Rout  = CartesianIndices(size(k)[1:end-1]) # indices for first n-1 dimensions
    M_out = length(Rout)
    M_in  = last(size(k))

    idx = CuArray{Int}(undef, size(k)...)
    r   = CuArray{Float32}(undef, last(size(k)), size(k)[1:end-1]...) # random numbers
    v   = CuArray{Float32}(undef, last(size(k)), size(k)[1:end-1]...) # inner likelihoods
    u   = CuArray{Float32}(undef, size(k)[1:end-1]...)                # outer likelihoods

    rng = GPUArrays.default_rng(CuArray)

    kernel  = @cuda launch=false kernel_likelihood_indices!(
                u, v,
                idx, k,
                model.q, model.σ,
                obs,
                r,
                rng.state,
                Rout, M_in
              )
    config  = launch_configuration(kernel.fun)
    threads = Base.min(M_out, config.threads, 256)
    blocks  = cld(M_out, threads)
    kernel(
        u, v,
        idx, k, 
        model.q, model.σ,
        obs,
        r,
        rng.state,
        Rout, M_in
        ;
        threads=threads, blocks=blocks
    )
    return u, idx
end

function likelihood_resample!(state::BinomialState, model, observation::BinomialObservation)
    u, idx = likelihood_indices(state.k, model, observation.EPSP)
    resample!(state.n, idx)
    resample!(state.k, idx)
    return u
end
