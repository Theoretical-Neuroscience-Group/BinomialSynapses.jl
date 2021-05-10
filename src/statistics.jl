MAP(fstate::NestedParticleState) = MAP(fstate.model)

function MAP(model::BinomialModel; marginal::Bool = false)
    if marginal
        return Dict(
            :N => mode(Array(model.N)),
            :p => mode(Array(model.p)),
            :q => mode(Array(model.q)),
            :σ => mode(Array(model.σ)),
            :τ => mode(Array(model.τ))
        )
    end
    v = hcat(
            model.N, 
            model.p, 
            model.q, 
            model.σ, 
            model.τ
        ) |> Array |> eachrow |> mode
    return Dict(
        :N => v[1],
        :p => v[2],
        :q => v[3],
        :σ => v[4],
        :τ => v[5]
    )
end

function MAP(model::BinomialGridModel)
    refresh!(model)
    MAP(BinomialModel(model.N, model.p, model.q, model.σ, model.τ))
end
