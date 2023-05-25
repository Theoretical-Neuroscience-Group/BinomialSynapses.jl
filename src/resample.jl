abstract type ResamplingMethod end
struct Multinomial <: ResamplingMethod end
struct Stratified <: ResamplingMethod end

function outer_indices!(u::AbstractVector, rm::ResamplingMethod)
    uu = Array(u)
    usum, idx = outer_indices!(uu, rm)
    return usum, cu(idx)
end

function outer_indices!(u::Vector, ::Multinomial)
    M_out  = length(u)
    usum   = zero(eltype(u))

    # compute cumulative sum, overwriting u
    @inbounds for i in 1:M_out
        usum += u[i]
        u[i] = usum
    end

    # shift cumulative sums to the right by one
    @inbounds for i in M_out:-1:2
        u[i] = u[i-1]
    end
    u[1] = 0

    # sample descending sequence of sorted random numbers
    # Algorithm by:
    # Bentley & Saxe, ACM Transactions on Mathematical Software, Vol 6, No 3
    # September 1980, Pages 359--364
    CurMax = one(eltype(u))
    idx = zeros(Int, M_out)
    bindex = M_out # bin index
    @inbounds for i in M_out:-1:1
        CurMax *= exp(log(rand(eltype(u))) / i)
        # scale random numbers (this is equivalent to normalizing u)
        rsample = CurMax * usum
        # checking bindex >= 1 is redundant since
        # ucum[1] = 0
        while rsample < u[bindex]
            bindex -= 1
        end
        idx[i] = bindex
    end
    return usum, idx
end

function outer_indices!(u::Vector, ::Stratified)
    M_out  = length(u)
    usum   = zero(eltype(u))

    # compute cumulative sum, overwriting u
    @inbounds for i in 1:M_out
        usum += u[i]
        u[i] = usum
    end

    # shift cumulative sums to the right by one
    @inbounds for i in M_out:-1:2
        u[i] = u[i-1]
    end
    u[1] = 0
    
    idx = zeros(Int, M_out)
    bindex = M_out # bin index
    @inbounds for i in M_out:-1:1
        rsample = (i - 1 + rand(eltype(u))) * usum / M_out
        # checking bindex >= 1 is redundant since
        # ucum[1] = 0
        while rsample < u[bindex]
            bindex -= 1
        end
        idx[i] = bindex
    end
    return usum, idx
end

"""
    outer_resample!(state, model, u)

Resample the outer particles of the `state` and `model` ensemble based on their likelihoods `u`.
"""
function outer_resample!(state, model, u, resampling_method)
    usum, idx = indices!(u, resampling_method)
    resample!(state, idx)
    resample!(model, idx)
    return state, model
end

"""
    indices!(v)

Return index table and total likelihoods from likelihood table `v`.
This function modifies v; after execution, v will be the cumulative sum of the original v
along the last dimension.
"""
indices!(v::AnyCuVector) = outer_indices!(v)
indices!(v::AnyCuVector, rm::ResamplingMethod) = outer_indices!(v, rm)
indices!(v::AnyCuVector, rm::Multinomial) = outer_indices!(v, rm)
indices!(v::AnyCuVector, rm::Stratified) = outer_indices!(v, rm)

function indices!(v::AnyCuArray, ::Multinomial)
    function kernel!(
        u, v, idx, r, 
        Rout, M_out, M_in
    )
        # grid-stride loop
        tid    = threadIdx().x
        window = (blockDim().x - 1i32) * gridDim().x
        offset = (blockIdx().x - 1i32) * blockDim().x
        while offset < M_out
            id = tid + offset
            # sample descending sequence of sorted random numbers
            # r[i,M_in] >= ... >= r[i,2] >= r[i,1]
            # Algorithm by:
            # Bentley & Saxe, ACM Transactions on Mathematical Software, Vol 6, No 3
            # September 1980, Pages 359--364
            vsum = 0f0
            CurMax = 1f0
            for j in 1:M_in
                mirrorj = M_in - j + 1 # mirrored index j
                CurMax *= CUDA.exp(CUDA.log(rand(Float32)) / mirrorj)
                if id <= M_out
                    @inbounds i = Rout[id]
                    if u[i] < 0 # prevents visiting same `i' more than once
                        # compute cumulative sums
                        @inbounds vsum = v[i, j] += vsum
                        @inbounds r[i, mirrorj] = CurMax
                    end
                end
            end
            if id <= M_out
                @inbounds i = Rout[id]

                if u[i] < 0 # prevents visiting same `i' more than once
                    # compute average likelihood across inner particles
                    # (with normalization constant that was omitted from v for speed)
                    @inbounds u[i] = vsum

                    # O(n) binning algorithm for sorted samples
                    bindex = 1 # bin index
                    for j in 1:M_in
                        # scale random numbers (this is equivalent to normalizing v)
                        @inbounds rsample = r[i, j] * vsum
                        # checking bindex <= M_in - 1 not necessary since
                        # v[i, M_in] = vsum
                        @inbounds while rsample > v[i, bindex]
                            bindex += 1
                        end
                        @inbounds idx[i, j] = bindex
                    end
                end
            end

            offset += window
        end
        return nothing
    end

    # initializations:

    # indices
    idx = CuArray{Int}(undef, size(v)...)   

    # outer likelihoods
    # Initialize to -1 in order to track which elements have been written to.
    # Since likelihoods are nonnegative, negative elements have never been visited.
    u   = CUDA.fill(-one(Float32), size(v)[1:end-1]...)     

    # random numbers
    r   = CuArray{Float32}(undef, size(v)...)

    Rout  = CartesianIndices(u) # indices for first n-1 dimensions
    M_out = length(u)
    M_in  = last(size(v))

    kernel  = @cuda launch=false kernel!(
                u, v, idx, r,
                Rout, M_out, M_in
              )
    config  = launch_configuration(kernel.fun)
    threads = max(32, min(config.threads, M_out))
    blocks  = cld(M_out, threads)
    kernel(
        u, v,
        idx, 
        r,
        Rout, M_out, M_in
        ;
        threads=threads, blocks=blocks
    )
    return u, idx
end

function indices!(v::AnyCuArray, ::Stratified)
    function kernel!(
        u, v, idx, r, 
        Rout, M_out, M_in
    )
        # grid-stride loop
        tid    = threadIdx().x
        window = (blockDim().x - 1i32) * gridDim().x
        offset = (blockIdx().x - 1i32) * blockDim().x
        while offset < M_out
            id = tid + offset
            vsum = 0f0
            for j in 1:M_in
                if id <= M_out
                    @inbounds i = Rout[id]
                    if u[i] < 0 # prevents visiting same `i' more than once
                        @inbounds vsum = v[i, j] += vsum
                        @inbounds r[i, j] = (j - 1 + rand(Float32)) / M_in
                    end
                end
            end
            if id <= M_out
                @inbounds i = Rout[id]

                if u[i] < 0 # prevents visiting same `i' more than once
                    # compute average likelihood across inner particles
                    # (with normalization constant that was omitted from v for speed)
                    @inbounds u[i] = vsum

                    # O(n) binning algorithm for sorted samples
                    bindex = 1 # bin index
                    for j in 1:M_in
                        # scale random numbers (this is equivalent to normalizing v)
                        @inbounds rsample = r[i, j] * vsum
                        # checking bindex <= M_in - 1 not necessary since
                        # v[i, M_in] = vsum
                        @inbounds while rsample > v[i, bindex]
                            bindex += 1
                        end
                        @inbounds idx[i, j] = bindex
                    end
                end
            end

            offset += window
        end
        return nothing
    end

    # initializations:

    # indices
    idx = CuArray{Int}(undef, size(v)...)   

    # outer likelihoods
    # Initialize to -1 in order to track which elements have been written to.
    # Since likelihoods are nonnegative, negative elements have never been visited.
    u = CUDA.fill(-one(Float32), size(v)[1:end-1]...)     

    # random numbers
    r = CuArray{Float32}(undef, size(v)...)

    Rout  = CartesianIndices(u) # indices for first n-1 dimensions
    M_out = length(u)
    M_in  = last(size(v))

    kernel  = @cuda launch=false kernel!(
                u, v, idx, r,
                Rout, M_out, M_in
              )
    config  = launch_configuration(kernel.fun)
    threads = max(32, min(config.threads, M_out))
    blocks  = cld(M_out, threads)
    kernel(
        u, v,
        idx, 
        r,
        Rout, M_out, M_in
        ;
        threads=threads, blocks=blocks
    )
    return u, idx
end

function resample!(in, out, idx)
    function kernel(in, out, idx, Ra, R1, R2, R3)
        i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        @inbounds if i <= length(in)
            I = Ra[i]     # choose high-level index
            I1 = R1[I[1]] # choose index before resampling dimension
            I2 = R2[I[2]] # choose index for resampling
            I3 = R3[I[3]] # choose index after resampling dimension
            out[I1, I2, I3] = in[I1, idx[I1, I2], I3]
        end#if
        return nothing
    end
    idx_dim = ndims(idx)
    R1 = CartesianIndices(size(in)[1:idx_dim-1]) # indices before resampling dimension
    R2 = CartesianIndices((size(in, idx_dim),)) # indices for resampling dimension
    R3 = CartesianIndices(size(in)[idx_dim+1:end]) # indices after resampling dimension

    Ra = CartesianIndices((length(R1), length(R2), length(R3))) # high-level indices

    kernel  = @cuda launch=false kernel(in, out, idx, Ra, R1, R2, R3)
    config  = launch_configuration(kernel.fun)
    threads = max(32, min(config.threads, length(out)))
    blocks  = cld(length(out), threads)
    kernel(in, out, idx, Ra, R1, R2, R3; threads=threads, blocks=blocks)
    return out
end

function resample!(in, idx)
    size(in)[1:ndims(idx)] == size(idx) || throw(DimensionMismatch("input and index array must have matching size"))
    out = similar(in)
    resample!(in, out, idx)
    in .= out
    return in
end

"""
    resample!(state, idx)
    resample!(model, idx)

Resample the outer particles of `state` or `model` ensembles based on index table `idx`.
"""
function resample!(state::BinomialState, idx)
    resample!(state.n, idx)
    resample!(state.k, idx)
    return state
end

function resample!(model::BinomialGridModel, idx)
    resample!(model.Nind, idx)
    resample!(model.pind, idx)
    resample!(model.qind, idx)
    resample!(model.σind, idx)
    resample!(model.τind, idx)
    return model
end
