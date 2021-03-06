using BinomialSynapses
using CUDA
using Distributions: Exponential

using BenchmarkTools
using Test

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

    include("_integration.jl")
end
