using BinomialSynapses
using CUDA

using BenchmarkTools
using Test

@testset "BinomialSynapses.jl" begin
    include("models.jl")
    include("propagate.jl")
    include("likelihood.jl")
    include("jitter.jl")
    include("resample.jl")
    include("filter.jl")
end
