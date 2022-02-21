# BinomialSynapses.jl

## Models
```@docs
AbstractBinomialModel
BinomialGridModel
BinomialGridModel(
    Nind, pind, qind, σind, τind,
    Nrng, prng, qrng, σrng, τrng
)
BinomialGridModel(m_out, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng)
BinomialModel
BinomialModel(Nmax, m_out, device = :gpu)
BinomialModel(m_out, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng)
BinomialModel(model::BinomialGridModel)
ScalarBinomialModel
BinomialState
BinomialState(Nmax, m_out, m_in, device = :gpu)
ScalarBinomialState(Nmax, device = :cpu)
ScalarBinomialState
BinomialObservation
propagate!(state::BinomialState, model, dt)
emit
```

## Time steps
```@docs
Timestep
FixedTimestep
RandomTimestep
get_step
```

## Particle Filter
```@docs
NestedParticleFilter
NestedParticleState
NestedParticleState(m_out, m_in, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng)
update!(
    filterstate::NestedParticleState,
    observation::BinomialObservation,
    filter::NestedParticleFilter
)
jitter!
likelihood
likelihood_resample!
outer_resample!
indices!
resample!
```

## OED
```@docs
OEDPolicy
policy
Uniform
MyopicPolicy
Myopic
MyopicFast
Myopic_tau
MyopicFast_tau
```

## Simulation
```@docs
NestedFilterSimulation
initialize!
propagate!(sim::NestedFilterSimulation)
run!
Recording
update!(rec::Recording, sim, time)
save
```


