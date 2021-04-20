function outer_indices!(u)
    uu = Array(u)
    idx = outer_indices!(uu)
    return cu(idx)
end

function outer_indices!(u::Vector)
    M_out  = length(u)
    usum   = 0f0
    # overwrite u with cumulative sum
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
        CurMax *= exp(log(rand()) / i)
        # scale random numbers (this is equivalent to normalizing u)
        rsample = CurMax * usum
        # checking bindex >= 1 is redundant since
        # ucum[1] = 0
        while rsample < u[bindex]
            bindex -= 1
        end
        idx[i] = bindex
    end
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
