function kernel_outer_indices!(idx, u, r, randstates)
    M_out = length(u)
    usum = u[M_out]
    CurMax = 1f0
    @inbounds for i in 1:M_out
        #usum += u[i]
        #u[i]  = usum
        # sample descending sequence of sorted random numbers
        # r[M_in] >= ... >= r[2] >= r[1]
        # Algorithm by:
        # Bentley & Saxe, ACM Transactions on Mathematical Software, Vol 6, No 3
        # September 1980, Pages 359--364
        mirrori    = M_out - i + 1 # mirrored index i
        CurMax    *= CUDA.exp(CUDA.log(GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates)) / mirrori)
        r[mirrori] = CurMax
    end
    bindex = 1 # bin index
    @inbounds for i in 1:M_out
        # scale random numbers (this is equivalent to normalizing u)
        rsample = r[i] * usum
        # checking bindex <= M_out - 1 is redundant since
        # u[M_in] = usum
        while rsample > u[bindex]
            bindex += 1
        end
        idx[i] = bindex
    end
    return nothing
end

function outer_indices!(u)
    M_out       = length(u)
    u          .= cumsum(u)
    idx         = CuArray{Int}(undef, M_out)
    r           = CuArray{Float32}(undef, M_out)

    rng = GPUArrays.default_rng(CuArray)

    kernel  = @cuda launch=false kernel_outer_indices!(idx, u, r, rng.state)
    config  = launch_configuration(kernel.fun)
    threads = Base.min(M_out, config.threads, 256)
    blocks  = cld(M_out, threads)
    kernel(idx, u, r, rng.state; threads=threads, blocks=blocks)
    return idx
end

function outer_resample!(state::BinomialState, model::BinomialGridModel, u)
    idx = outer_indices!(u)
    state.n .= state.n[idx,:]
    state.k .= state.k[idx,:]
    model.Nind .= model.Nind[idx]
    model.pind .= model.pind[idx]
    model.qind .= model.qind[idx]
    model.σind .= model.σind[idx]
    model.τind .= model.τind[idx]
    return state, model
end
