# BinomialSynapses

[![Build status](https://badge.buildkite.com/15db27ead6ca652df308f96b4805115a1720f1d75155d90b63.svg)](https://buildkite.com/theoretical-neuroscience-group/binomialsynapses)
[![Coverage](https://codecov.io/gh/Theoretical-Neuroscience-Group/BinomialSynapses.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Theoretical-Neuroscience-Group/BinomialSynapses.jl)

## Installation

To install this package in Julia 1.5 or 1.6, type

```julia
]add https://github.com/Theoretical-Neuroscience-Group/BinomialSynapses.jl
```

## Usage

User API is work in progress. This is a minimal working example for a basic setup and performing a single update of the nested particle filter:
```julia
using BinomialSynapses, CUDA

state = BinomialState(128, 1024, 1024)
model = BinomialGridModel(
    1024,
    1:5,
    LinRange(0.05,0.95,5),
    LinRange(0.1,2,5),
    LinRange(0.05,2,5),
    LinRange(0.05,2,5)
)

fstate = NestedParticleState(state, model)
filter = NestedParticleFilter(12)
obs    = BinomialObservation(0.3f0, 0.1f0)

update!(fstate, obs, filter)
```
