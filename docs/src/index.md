# BinomialSynapses.jl

## Models
```@docs
AbstractBinomialModel
BinomialModel
BinomialGridModel
ScalarBinomialModel
BinomialState
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
update!
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
run!
Recording
update!(rec::Recording, sim, time)
save
```


