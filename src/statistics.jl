using Statistics

MAP(fstate::NestedParticleState) = MAP(fstate.model)
variance(fstate::NestedParticleState) = variance(fstate.model)

function MAP(model::BinomialModel)
        Dict(
                :N => mode(Array(model.N)),
                :p => mode(Array(model.p)),
                :q => mode(Array(model.q)),
                :σ => mode(Array(model.σ)),
                :τ => mode(Array(model.τ))
        )
end

function variance(model::BinomialModel)
        Dict(
                :N => var(Array(model.N)),
                :p => var(Array(model.p)),
                :q => var(Array(model.q)),
                :σ => var(Array(model.σ)),
                :τ => var(Array(model.τ))
        )
end

function MAP(model::BinomialGridModel)
        refresh!(model)
        MAP(BinomialModel(model.N, model.p, model.q, model.σ, model.τ))
end

function variance(model::BinomialGridModel)
        refresh!(model)
        variance(BinomialModel(model.N, model.p, model.q, model.σ, model.τ))
end
