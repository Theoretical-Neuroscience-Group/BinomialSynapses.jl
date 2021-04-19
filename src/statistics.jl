MAP(fstate::NestedParticleState) = MAP(fstate.model)

function MAP(model::BinomialModel)
        Dict(
                :N => mode(Array(model.N)),
                :p => mode(Array(model.p)),
                :q => mode(Array(model.q)),
                :σ => mode(Array(model.σ)),
                :τ => mode(Array(model.τ))
        )
end

function MAP(model::BinomialGridModel)
        refresh!(model)
        MAP(BinomialModel(model.N, model.p, model.q, model.σ, model.τ))
end
