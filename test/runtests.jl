using BinomialSynapses
using CUDA

using BenchmarkTools
using Test

@testset "BinomialSynapses.jl" begin
    include("types.jl")
    include("propagate.jl")
    include("likelihood.jl")
    include("update_parameters.jl")
    include("resample.jl")
end
