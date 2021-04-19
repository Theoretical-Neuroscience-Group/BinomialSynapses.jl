module BinomialSynapses

using BinomialGPU
using CUDA
using Distributions: Binomial, Exponential, Normal
using GPUArrays
using LaTeXStrings
using Plots
using Statistics: mean

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

include("visualize.jl")
export posterior_plot

end#module
