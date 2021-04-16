# BinomialSynapses

[![Build status](https://badge.buildkite.com/15db27ead6ca652df308f96b4805115a1720f1d75155d90b63.svg)](https://buildkite.com/theoretical-neuroscience-group/binomialsynapses)
[![Coverage](https://codecov.io/gh/Theoretical-Neuroscience-Group/BinomialSynapses.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Theoretical-Neuroscience-Group/BinomialSynapses.jl)

## Installation

To install this package in Julia 1.5 or 1.6, type

```julia
]add https://github.com/Theoretical-Neuroscience-Group/BinomialSynapses.jl
```

## Usage

User API is work in progress. This is a sketch for the function performing a single update of the nested particle filter:
```julia
function update!(filterstate::NestedFilterState, observation::BinomialObservation, filter::NestedFilter)
    state = filterstate.state
    model = filterstate.model

    update!(model, filter.jittering_width)
    propagate!(state, model, observation.dt)
    u = likelihood_resample!(state, model, observation)
    outer_resample!(state, u)
    return filterstate
end
```
