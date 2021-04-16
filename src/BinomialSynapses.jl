module BinomialSynapses

using BinomialGPU
using CUDA
using GPUArrays
using Statistics: mean

# data structures and types
include("types.jl")
export BinomialModel
export BinomialState

# filtering part
include("propagate.jl")
include("likelihood.jl")
include("resample.jl")
export
        likelihood,
        likelihood_resample!,
        propagate!,
        outer_resample!

end#module
