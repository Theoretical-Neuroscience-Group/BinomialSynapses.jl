module BinomialSynapses

using BinomialGPU
using CUDA
using Statistics: mean

# data structures and types
include("types.jl")
export BinomialModel

# filtering part
include("propagate.jl")
include("likelihood.jl")
export
        likelihood,
        likelihood_resample!,
        propagate!

end#module
