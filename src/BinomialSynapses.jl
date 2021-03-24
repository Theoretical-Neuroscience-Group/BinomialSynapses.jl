module BinomialSynapses

using BinomialGPU
using CUDA

# data structures and types
include("types.jl")
export BinomialModel

# filtering part
include("propagate.jl")
include("likelihood.jl")
export likelihood, propagate!

end
