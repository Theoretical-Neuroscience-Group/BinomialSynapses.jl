# Same algorithm as in https://tianjun.me/essays/Categorical_Sampling_on_GPU_with_Julia
function make_alias_table!(w::AbstractVector{T}, wsum::S,
                           a::AbstractVector{T},
                           alias::AbstractVector{<:Integer}) where {S, T}
    n = length(w)
    length(a) == length(alias) == n ||
        throw(DimensionMismatch("Inconsistent array lengths."))

    ac = n / wsum
    for i = 1:n
        @inbounds a[i] = w[i] * ac
    end

    larges = Vector{Int}(undef, n)
    smalls = Vector{Int}(undef, n)
    kl = 0  # actual number of larges
    ks = 0  # actual number of smalls

    for i = 1:n
        @inbounds ai = a[i]
        if ai > 1.0
            larges[kl+=1] = i  # push to larges
        elseif ai < 1.0
            smalls[ks+=1] = i  # push to smalls
        end
    end

    while kl > 0 && ks > 0
        s = smalls[ks]; ks -= 1  # pop from smalls
        l = larges[kl]; kl -= 1  # pop from larges
        @inbounds alias[s] = l
        @inbounds al = a[l] = (a[l] - 1.0) + a[s]
        if al > 1.0
            larges[kl+=1] = l  # push to larges
        else
            smalls[ks+=1] = l  # push to smalls
        end
    end

    # this loop should be redundant, except for rounding
    for i = 1:ks
        @inbounds a[smalls[i]] = 1.0
    end
    nothing
end

function cu_alias_sample!(a::CuArray{Ta}, wv::AbstractVector{Tw}, x::CuArray{Ta}) where {Tw<:Number, Ta}
    length(a) == length(wv) || throw(DimensionMismatch("weight vector must have the same length with label vector"))
    n = length(wv)
    # create alias table
    ap = Vector{Tw}(undef, n)
    alias = Vector{Int64}(undef, n)
    make_alias_table!(wv, sum(wv), ap, alias)

    # to device
    alias = CuArray{Int64}(alias)
    ap = CuArray{Tw}(ap)

    function kernel(state, _, (alias, ap, x, a, randstate))
        r1, r2 = GPUArrays.gpu_rand(Float32, state, randstate), GPUArrays.gpu_rand(Float32, state, randstate)
        r1 = r1 == 1.0 ? 0.0 : r1
        r2 = r2 == 1.0 ? 0.0 : r2
        i = linear_index(state)
        if i <= length(x)
            j = floor(Int, r1 * n) + 1
            @inbounds x[i] = r2 < ap[j] ? a[j] : a[alias[j]]
        end
        return
    end
    gpu_call(kernel, x, (alias, ap, x, a, GPUArrays.default_rng(typeof(x)).state))
    x
end

function outer_resample!(state::BinomialState, u)
    M_out = size(state.n, 1)
    a = CuArray{Int64}(1:M_out)
    x = CuArray{Int64}(zeros(Int64, M_out))
    cu_alias_sample!(a,u,x)
    state.n .= state.n[x,:]
    state.k .= state.k[x,:]
end
