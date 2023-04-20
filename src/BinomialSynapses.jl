module BinomialSynapses

using BinomialGPU
using CUDA
using CUDA: i32
using Distributions: Binomial, Exponential, Normal
using LaTeXStrings
using Plots
using Statistics: mean, cov, cor
using StatsBase: mode, entropy
using LinearAlgebra: det
using CovarianceEstimation

include("models.jl")
export
       AbstractBinomialModel,
       BinomialModel,
       BinomialGridModel,
       ScalarBinomialModel,
       BinomialState,
       ScalarBinomialState,
       BinomialObservation

include("propagate.jl")
export propagate!

include("timestep.jl")
export Timestep, FixedTimestep, RandomTimestep, get_step, DeterministicTrain, BatchTrain

include("emission.jl")
export emit

include("likelihood.jl")
export likelihood, likelihood_resample!

include("jitter.jl")
export jitter!

include("resample.jl")
export outer_resample!, indices!, resample!

include("filter.jl")
export NestedParticleFilter, NestedParticleState, update!

include("experiment.jl")
export NestedFilterExperiment, run_experiment!, postpro_experiment!

include("statistics.jl")
export MAP, MEAN

include("record.jl")
export Recording

include("simulate.jl")
export NestedFilterSimulation, initialize!, m_out, run!, runBatch_map!, runBatchTau!, run_exact!

include("visualize.jl")
export posterior_plot

include("OED.jl")
export OEDPolicy, policy, Uniform

include("myopic.jl")
export MyopicPolicy, Myopic, MyopicFast, Myopic_tau, MyopicFast_tau

end#module
