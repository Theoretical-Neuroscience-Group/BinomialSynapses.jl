module BinomialSynapses

using BinomialGPU
using CUDA
using GPUArrays
using Statistics: mean

include("models.jl")
        export
               BinomialModel,
               BinomialGridModel,
               BinomialState,
               BinomialObservation

include("propagate.jl")
export propagate!

include("likelihood.jl")
export likelihood, likelihood_resample!

include("jitter.jl")
export jitter!

include("resample.jl")
export outer_resample!

include("filter.jl")
export NestedParticleFilter, NestedParticleState, update!

end#module
