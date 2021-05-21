MAP(fstate::NestedParticleState; kwargs...) = MAP(fstate.model, kwargs...)

function MAP(model::BinomialGridModel; kwargs...)
    MAP(BinomialModel(model); kwargs...)
end

function MAP(model::BinomialModel{T1, T2}; marginal::Bool = false) where {T1, T2}
    if marginal
        return BinomialModel(
            mode(Array(model.N)),
            mode(Array(model.p)),
            mode(Array(model.q)),
            mode(Array(model.σ)),
            mode(Array(model.τ))
        )
    end
    v = hcat(
            model.N, 
            model.p, 
            model.q, 
            model.σ, 
            model.τ
        ) |> Array |> eachrow |> mode
    return BinomialModel{eltype(T1), eltype(T2)}(eltype(T1)(v[1]), v[2:end]...)
end
