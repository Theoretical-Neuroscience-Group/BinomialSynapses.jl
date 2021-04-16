module BinomialSynapses

using BinomialGPU
using CUDA
using GPUArrays

# data structures and types
include("types.jl")
export BinomialModel

# filtering part
include("propagate.jl")
include("likelihood.jl")
export likelihood, propagate!

end
