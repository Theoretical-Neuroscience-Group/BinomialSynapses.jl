module BinomialSynapses

using BinomialGPU
using CUDA
using GPUArrays
using Statistics: mean

# data structures and types
include("types.jl")
export BinomialModel

# filtering part
include("propagate.jl")
include("likelihood.jl")
        likelihood,
        likelihood_resample!,
        propagate!
include("resample.jl")
        outer_resample!

end#module
