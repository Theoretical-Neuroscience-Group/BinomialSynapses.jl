# BinomialSynapses

[![Build status](https://badge.buildkite.com/15db27ead6ca652df308f96b4805115a1720f1d75155d90b63.svg)](https://buildkite.com/theoretical-neuroscience-group/binomialsynapses)
[![Coverage](https://codecov.io/gh/Theoretical-Neuroscience-Group/BinomialSynapses.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Theoretical-Neuroscience-Group/BinomialSynapses.jl)

## Installation

To install this package in Julia 1.5 or 1.6, type

```julia
]add https://github.com/Theoretical-Neuroscience-Group/BinomialSynapses.jl
```

## Usage

User API is work in progress. This is a minimal working example for running the nested particle filter on synthetic data:
```julia
using BinomialSynapses

hidden = ScalarBinomialState(125, 62)
hmodel = ScalarBinomialModel(128, 0.1, 0.2, 0.3, 0.4)
fstate = NestedParticleState(16, 16, 1:5, LinRange(0.05,0.95,5), LinRange(0.1,2,5), LinRange(0.05,2,5), LinRange(0.05,2,5))
filter = NestedParticleFilter(12)

T = 1000
for i in 1:T
    println("t= ", i)
    obs = propagate_emit!(hidden, hmodel)
    update!(fstate, obs, filter)
    # add output
end
```
