using BinomialSynapses
using CUDA
using Distributions: Exponential
using StatsBase: entropy, proportionmap

using BenchmarkTools
using Test

CUDA.versioninfo()

const RUN_BENCHMARKS = false # optional intermediate benchmarks

@testset "BinomialSynapses.jl" begin
    include("models.jl")
    include("propagate.jl")
    include("timestep.jl")
    include("emission.jl")
    include("likelihood.jl")
    include("jitter.jl")
    include("resample.jl")
    include("filter.jl")
    include("statistics.jl")
    include("simulate.jl")
    include("record.jl")
    include("myopic.jl")

    include("_integration.jl")

    include("debug.jl")
end
