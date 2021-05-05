module BinomialSynapses

using BinomialGPU
using CUDA
using Distributions: Binomial, Exponential, Normal
using LaTeXStrings
using Plots
using Random
using Statistics: mean
using StatsBase: mode

include("models.jl")
export
       AbstractBinomialModel,
       BinomialModel,
       BinomialGridModel,
       ScalarBinomialModel,
       BinomialState,
       ScalarBinomialState,
       BinomialObservation

include("propagate.jl")
export propagate!

include("emission.jl")
export propagate_emit!

include("likelihood.jl")
export likelihood, likelihood_resample!

include("jitter.jl")
export jitter!

include("resample.jl")
export outer_resample!

include("filter.jl")
export NestedParticleFilter, NestedParticleState, update!

include("statistics.jl")
export MAP

include("simulate.jl")
export NestedFilterSimulation, run!

include("visualize.jl")
export posterior_plot

end#module
