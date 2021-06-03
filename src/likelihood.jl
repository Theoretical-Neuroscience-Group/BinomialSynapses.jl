function likelihood(k, model::AbstractBinomialModel, observation)
    return mean(
                exp.(-0.5f0 .* ((observation .- model.q .* k) ./ model.σ).^2)
                ./ (sqrt(2*Float32(pi)) .* model.σ)
           , dims = 2
           )[:,1]
end

function kernel_likelihood_indices!(u, v, idxT, kT, q, σ, observation, r, randstates)
    # use column-index first (gives a 3x speedup)
    # kT stands for the transpose of k
    # idxT stands for the transpose of idx
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    M_in, M_out = size(kT)
    @inbounds if j <= M_out
        vsum = 0f0
        CurMax = 1f0
        for i in 1:M_in
            # omitting normalization constant here; it is only needed for u
            vi      = CUDA.exp(-0.5f0 *((observation - q[j] * kT[i,j]) / σ[j])^2)
            vsum   += vi
            v[i, j] = vsum
            # sample descending sequence of sorted random numbers
            # r[i,M_in] >= ... >= r[i,2] >= r[i,1]
            # Algorithm by:
            # Bentley & Saxe, ACM Transactions on Mathematical Software, Vol 6, No 3
            # September 1980, Pages 359--364
            mirrori = M_in - i + 1 # mirrored index i
            CurMax *= CUDA.exp(CUDA.log(GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates)) / mirrori)
            r[mirrori, j] = CurMax
        end
        # compute average likelihood across inner particles
        # (with normalization constant that was omitted from v for speed)
        u[j] = vsum / (M_in * CUDA.sqrt(2*Float32(pi)) * σ[j])
        # O(n) binning algorithm for sorted samples
        bindex = 1 # bin index
        @inbounds for i in 1:M_in
            # scale random numbers (this is equivalent to normalizing v)
            rsample = r[i, j] * vsum
            # checking bindex <= M_in - 1 is redundant since
            # v[M_in, j] = vsum
            while rsample > v[bindex, j]
                bindex += 1
            end
            idxT[i, j] = bindex
        end
    end
    return nothing
end

function likelihood_indices(k, model::AbstractBinomialModel, observation)
    M_out, M_in = size(k)
    r           = CuArray{Float32}(undef, M_in, M_out)
    u           = CuArray{Float32}(undef, M_out)
    v           = CuArray{Float32}(undef, M_in, M_out)
    idx         = CuArray{Int}(undef, M_out, M_in)

    rng = GPUArrays.default_rng(CuArray)

    kernel  = @cuda launch=false kernel_likelihood_indices!(
                u, v,
                idx', k',
                model.q, model.σ,
                Float32(observation),
                r,
                rng.state
              )
    config  = launch_configuration(kernel.fun)
    threads = Base.min(M_out, config.threads, 256)
    blocks  = cld(M_out, threads)
    kernel(
        u, v,
        idx', k', # pass transposes for column-major indexing (gives a 3x speedup)
        model.q, model.σ,
        observation,
        r,
        rng.state
        ;
        threads=threads, blocks=blocks
    )
    return u, idx
end

function inner_resample_helper!(in, out, idx)
    function kernel(in, out, idx, R)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        @inbounds if i <= length(in)
            I = R[i]
            out[I] = in[I[1], idx[I]]
        end#if
        return nothing
    end
    R = CartesianIndices(size(in))

    kernel  = @cuda launch=false kernel(in, out, idx, R)
    config  = launch_configuration(kernel.fun)
    threads = Base.min(length(out), config.threads, 256)
    blocks  = cld(length(out), threads)
    kernel(in, out, idx, R; threads=threads, blocks=blocks)
    return out
end

function inner_resample_helper!(in, idx)
    size(in) == size(idx) || throw(DimensionMismatch("input and index array must have the same size"))
    out = similar(in)
    inner_resample_helper!(in, out, idx)
    in .= out
    return in
end

function likelihood_resample!(state::BinomialState, model, observation::BinomialObservation)
    u, idx = likelihood_indices(state.k, model, observation.EPSP)
    inner_resample_helper!(state.n, idx)
    inner_resample_helper!(state.k, idx)
    return u
end
