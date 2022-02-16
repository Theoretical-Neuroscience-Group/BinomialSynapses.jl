function show_histogram(
    bins, indices;
    truevalue = nothing, xlabel = L"N [-]", ylabel = L"p(N)"
)
    nbins = length(bins)
    counts = zeros(nbins)
    for i in 1:length(indices)
        counts[indices[i]] += 1
    end
    p = plot(
            bins,
            counts./sum(counts),
            xlabel=xlabel,
            ylabel=ylabel,
            legend=false
           )
    if !isnothing(truevalue)
        plot!([truevalue], seriestype="vline",legend=false)
    end
    return p
end

function show_EPSP_trace(times, epsps)
    return plot(
        times,
        epsps,
        xlabel="Time [s]",
        ylabel="EPSC [A]",
        legend=false
    )
end

function flatten(A::AbstractMatrix)
    n,m = size(A)
    return reshape(A, n*m)
end

function posterior_plot(
    fstate, times, epsps;
    truemodel = nothing,
    truestate = nothing,
    showstates = false
)
    Nrng = Array(fstate.model.Nrng)
    prng = Array(fstate.model.prng)
    qrng = Array(fstate.model.qrng)
    σrng = Array(fstate.model.σrng)
    τrng = Array(fstate.model.τrng)

    Nind = Array(fstate.model.Nind)
    pind = Array(fstate.model.pind)
    qind = Array(fstate.model.qind)
    σind = Array(fstate.model.σind)
    τind = Array(fstate.model.τind)

    pE = show_EPSP_trace(times, epsps)
    pN = show_histogram(Nrng, Nind,
            xlabel = L"N [-]", ylabel = L"p(N)")
    !isnothing(truemodel) && plot!([truemodel.N[1]], seriestype="vline",legend=false)
    pp = show_histogram(prng, pind,
            xlabel = L"p [-]", ylabel = L"p(p)")
    !isnothing(truemodel) && plot!([truemodel.p[1]], seriestype="vline",legend=false)
    pq = show_histogram(qrng, qind,
            xlabel = L"q [A]", ylabel = L"p(q)")
    !isnothing(truemodel) && plot!([truemodel.q[1]], seriestype="vline",legend=false)
    pσ = show_histogram(σrng, σind,
            xlabel = L"\sigma [A]", ylabel = L"p(\sigma)")
    !isnothing(truemodel) && plot!([truemodel.σ[1]], seriestype="vline",legend=false)
    pτ = show_histogram(τrng, τind,
            xlabel = L"\tau [s]", ylabel = L"p(\tau)")
    !isnothing(truemodel) && plot!([truemodel.τ[1]], seriestype="vline",legend=false)
    if showstates
        pn = histogram(flatten(Array(fstate.state.n)), bins=1:20,
                normalize = :probability,
                xlabel = L"n", ylabel = L"p(n)")
        !isnothing(truestate) && plot!([truestate.n[1,1]], seriestype="vline",legend=false)
        pk = histogram(flatten(Array(fstate.state.k)), bins=1:20,
                normalize = :probability,
                xlabel = L"k", ylabel = L"p(k)")
        !isnothing(truestate) && plot!([truestate.k[1,1]], seriestype="vline",legend=false)
        display(plot(pE, pN, pp, pq, pσ, pτ, pn, pk, layout = (4, 2)))
        return
    end
    display(plot(pE, pN, pp, pq, pσ, pτ, layout = (3, 2)))
end

function posterior_plot(sim::NestedFilterSimulation)
    return posterior_plot(sim.fstate, sim.times, sim.epsps, truemodel = sim.hmodel)
end
