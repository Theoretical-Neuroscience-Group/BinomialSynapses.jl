struct BinomialModel{T1,T2}
    N::T1
    p::T2
    q::T2
    sigma::T2
    tau::T2
end

struct BinomialState{T}
    n::T
    k::T
end
