module BinomialSynapses

using BinomialGPU
using CUDA

# data structures and types
include("types.jl")
export
       BinomialModel,
       BinomialGridModel

# filtering part
include("propagate.jl")
include("likelihood.jl")
include("update_parameters.jl")
export
        likelihood,
        propagate!,
        update!

end
