function outer_indices!(u::AbstractVector)
    uu = Array(u)
    usum, idx = outer_indices!(uu)
    return usum, cu(idx)
end

function outer_indices!(u::Vector)
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

function outer_resample!(state, model, u)
    usum, idx = outer_indices!(u)
    resample!(state, idx)
    resample!(model, idx)
    return state, model
end

indices!(v::AnyCuVector) = outer_indices!(v)

# produce index table and total likelihoods from likelihood table
# (this function modifies v; after execution, v will be the cumulative sum of the original v
# along the last dimension)
function indices!(v::AnyCuArray)
    function kernel!(
        u, v, idx, r, 
        randstates, 
        Rout, M_out, M_in
    )
        id = (blockIdx().x - 1) * blockDim().x + threadIdx().x # physical index
        @inbounds if id <= M_out
            i = Rout[id]
            vsum = 0f0
            CurMax = 1f0
            @inbounds for j in 1:M_in
                # compute cumulative sums
                vsum = v[i, j] += vsum
                # sample descending sequence of sorted random numbers
                # r[M_in,i] >= ... >= r[2,i] >= r[1,i]
                # Algorithm by:
                # Bentley & Saxe, ACM Transactions on Mathematical Software, Vol 6, No 3
                # September 1980, Pages 359--364
                mirrorj = M_in - j + 1 # mirrored index i
                CurMax *= CUDA.exp(CUDA.log(GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates)) / mirrorj)
                r[i, mirrorj] = CurMax
            end
            # compute average likelihood across inner particles
            # (with normalization constant that was omitted from v for speed)
            u[i] = vsum 
            # O(n) binning algorithm for sorted samples
            bindex = 1 # bin index
            @inbounds for j in 1:M_in
                # scale random numbers (this is equivalent to normalizing v)
                rsample = r[i, j] * vsum
                # checking bindex <= M_in - 1 is redundant since
                # v[M_in, j] = vsum
                while rsample > v[i, bindex]
                    bindex += 1
                end
                idx[i, j] = bindex
            end
        end
        return nothing
    end

    idx = CuArray{Int}(undef, size(v)...)              # indices
    u   = CuArray{Float32}(undef, size(v)[1:end-1]...) # outer likelihoods
    r   = CuArray{Float32}(undef, size(v)...)          # random numbers

    Rout  = CartesianIndices(u) # indices for first n-1 dimensions
    M_out = length(u)
    M_in  = last(size(v))

    rng = GPUArrays.default_rng(CuArray)

    kernel  = @cuda launch=false kernel!(
                u, v, idx, r,
                rng.state,
                Rout, M_out, M_in
              )
    config  = launch_configuration(kernel.fun)
    threads = Base.min(M_out, config.threads, 256)
    blocks  = cld(M_out, threads)
    kernel(
        u, v,
        idx, 
        r,
        rng.state,
        Rout, M_out, M_in
        ;
        threads=threads, blocks=blocks
    )
    return u, idx
end

function resample!(in, out, idx)
    function kernel(in, out, idx, Ra, R1, R2, R3)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
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
    threads = Base.min(length(out), config.threads, 256)
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
