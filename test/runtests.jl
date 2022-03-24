using BinomialSynapses
using CUDA
using Distributions: Exponential
using JLD
using StatsBase

using BenchmarkTools
using Test

const DEVICES = [:cpu]

if CUDA.functional()
    @info "Functional CUDA device detected."
    CUDA.versioninfo()
    push!(DEVICES, :gpu)
else
    @warn "No CUDA device detected. Skipping GPU tests."
end

const RUN_BENCHMARKS = false # optional intermediate benchmarks
if RUN_BENCHMARKS
    @info "Running full suite of benchmarks."
else
    @info "Skipping intermediate benchmarks."
end

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

    # include("_integration.jl")
    include("saving.jl")
end
