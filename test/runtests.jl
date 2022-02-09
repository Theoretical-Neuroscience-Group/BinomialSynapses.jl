using BinomialSynapses
using CUDA
using StatsBase: proportionmap, entropy

using BenchmarkTools
using Test

const RUN_BENCHMARKS = false # optional intermediate benchmarks

@testset "BinomialSynapses.jl" begin
    # include("models.jl")
    # include("propagate.jl")
    # include("emission.jl")
    # include("likelihood.jl")
    # include("jitter.jl")
    # include("resample.jl")
    # include("filter.jl")
    # include("statistics.jl")

    # include("_integration.jl")

    include("debug.jl")
end
