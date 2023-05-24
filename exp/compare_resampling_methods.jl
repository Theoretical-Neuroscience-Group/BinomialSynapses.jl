using BinomialSynapses, Plots
using Statistics

function f1(sim, time)
    return abs2(mean(sim.fstate.model.N) - only(sim.hmodel.N)),
        abs2(mean(sim.fstate.model.p) - only(sim.hmodel.p)),
        abs2(mean(sim.fstate.model.q) - only(sim.hmodel.q)),
        abs2(mean(sim.fstate.model.τ) - only(sim.hmodel.τ)),
        abs2(mean(sim.fstate.model.σ) - only(sim.hmodel.σ))
end

f2(data) = nothing

function make_sim(rm)
    return NestedFilterSimulation(
        10, 0.85, 1.0, 0.2, 0.2,   # ground truth parameters
        1:20,                      # parameter ranges for filter
        LinRange(0.05, 0.95, 100), # .
        LinRange(0.10, 2.00, 100), # .
        LinRange(0.05, 2.00, 100), # .
        LinRange(0.05, 2.00, 100), # .
        2048, 512,                 # outer and inner number of particles
        12;                        # jittering kernel width
        resampling_method = rm
    )
end

function run_and_record!(rec, rm)
    sim = make_sim(rm)
    run!(sim; T = 1000, recording = rec)
    return rec
end

function run_and_record(rm)
    sim = make_sim(rm)
    rec = Recording(f1, f2, sim)
    run!(sim; T = 1000, recording = rec)
    return rec
end

rec1 = run_and_record(Multinomial())
rec2 = run_and_record(Stratified())

for i in 1:99
    run_and_record!(rec1, Multinomial())
    run_and_record!(rec2, Stratified())
end

pop!(rec1.data)
pop!(rec2.data)

dat1 = reduce(hcat, mean(collect, reshape(rec1.data, 1000, 100); dims=2)[:, 1])
dat2 = reduce(hcat, mean(collect, reshape(rec2.data, 1000, 100); dims=2)[:, 1])

names = ["N", "p", "q", "τ", "σ"]
for (i, name) in enumerate(names)
    plt = plot(; ylabel = "E(Estimated $name - True $name)²", xlabel = "Time")
    plot!(plt, dat1[i, :]; label = "Multinomial resampling")
    plot!(plt, dat2[i, :]; label = "Stratified resampling")
    savefig(plt, "exp/resampling_$name.svg")
end
