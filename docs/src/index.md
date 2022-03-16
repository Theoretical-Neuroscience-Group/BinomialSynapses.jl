# BinomialSynapses.jl

## Models
```@docs
AbstractBinomialModel
BinomialGridModel
BinomialGridModel(
    Nind, pind, qind, σind, τind,
    Nrng, prng, qrng, σrng, τrng
)
BinomialGridModel(m_out::Integer, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng)
BinomialModel
BinomialModel(Nmax::Integer, m_out::Integer, device::Symbol = :gpu)
BinomialModel(m_out::Integer, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng)
BinomialModel(model::BinomialGridModel)
ScalarBinomialModel
BinomialState(Nmax::Integer, m_out::Integer, m_in::Integer, device::Symbol = :gpu)
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
NestedParticleState(
    m_out::Integer, m_in::Integer,
    my_Nrng, my_prng, my_qrng, my_σrng, my_τrng
)
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
```


