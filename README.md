# BinomialSynapses
[![ci](https://github.com/Theoretical-Neuroscience-Group/BinomialSynapses.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/Theoretical-Neuroscience-Group/BinomialSynapses.jl/actions/workflows/ci.yaml)
[![Codecov](https://codecov.io/gh/Theoretical-Neuroscience-Group/BinomialSynapses.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Theoretical-Neuroscience-Group/BinomialSynapses.jl)
[![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://theoretical-neuroscience-group.github.io/BinomialSynapses.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://theoretical-neuroscience-group.github.io/BinomialSynapses.jl/dev/
<!-- [![Build status](https://badge.buildkite.com/15db27ead6ca652df308f96b4805115a1720f1d75155d90b63.svg)](https://buildkite.com/theoretical-neuroscience-group/binomialsynapses) -->
<!-- [![Coverage](https://codecov.io/gh/Theoretical-Neuroscience-Group/BinomialSynapses.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Theoretical-Neuroscience-Group/BinomialSynapses.jl) -->

This is a package for performing filering and active learning for a binomial synaptic model using nested particle filters.
Performance is achieved by providing a CUDA GPU implementation, but the code also runs (much more slowly) on the CPU.

## Installation

This package needs at least Julia 1.6.1. 
In a Julia REPL, activate an environment and type:

```julia
]add BinomialSynapses
```

## Usage

User API is work in progress. This is a minimal working example for running the nested particle filter on synthetic data and producing a plot of the observation trace and the posterior histograms.

```julia
using BinomialSynapses

sim = NestedFilterSimulation(
        10, 0.85, 1.0, 0.2, 0.2,   # ground truth parameters
        1:20,                      # parameter ranges for filter
        LinRange(0.05, 0.95, 100), # .
        LinRange(0.10, 2.00, 100), # .
        LinRange(0.05, 2.00, 100), # .
        LinRange(0.05, 2.00, 100), # .
        2048, 512,                 # outer and inner number of particles
        12                         # jittering kernel width
      )

times, epsps = run!(sim, T = 1000)

posterior_plot(sim.fstate, times, epsps, truemodel = sim.hmodel)
```

![](posteriors.png)


## References

- On the nested particle filter: Crisan, Dan, and Joaquin Miguez. "Nested particle filters for online parameter estimation in discrete-time state-space Markov models." Bernoulli 24.4A (2018): 3039-3086.
- On the model of stochastic synapse: Gontier, Camille, and Jean-Pascal Pfister. "Identifiability of a Binomial Synapse." Frontiers in computational neuroscience 14 (2020): 86.

## Citing this work
Please cite the following paper:

[Camille Gontier, Simone Carlo Surace, Igor Delvendahl, Martin Müller, Jean-Pascal Pfister (2022). Efficient Sampling-Based Bayesian Active Learning for synaptic characterization. arXiv preprint.
](https://arxiv.org/abs/2201.07539)
