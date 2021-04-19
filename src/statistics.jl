MAP(fstate::NestedParticleState) = MAP(fstate.model)

function MAP(model::BinomialModel)
        Dict(
                :N => mode(Array(model.N)),
                :p => mode(Array(model.p)),
                :q => mode(Array(model.q)),
                :σ => mode(Array(model.sigma)),
                :τ => mode(Array(model.tau))
        )
end

function MAP(model::BinomialGridModel)
        refresh!(model)
        MAP(BinomialModel(model.N, model.p, model.q, model.sigma, model.tau))
end
