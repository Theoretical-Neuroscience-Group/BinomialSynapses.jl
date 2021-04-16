module BinomialSynapses

using BinomialGPU
using CUDA
using GPUArrays
using Statistics: mean

# data structures and types
include("types.jl")

export
       BinomialModel,
       BinomialGridModel,
       BinomialState

# filtering part
include("propagate.jl")
include("likelihood.jl")
include("update_parameters.jl")
include("resample.jl")
export
        likelihood,
        likelihood_resample!,
        propagate!,
        update!,
        outer_resample!

end#module
