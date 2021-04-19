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
    if truevalue != nothing
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
    Nrng     = Array(fstate.model.Nrng)
    prng     = Array(fstate.model.prng)
    qrng     = Array(fstate.model.qrng)
    sigmarng = Array(fstate.model.sigmarng)
    taurng   = Array(fstate.model.taurng)

    Nind     = Array(fstate.model.Nind)
    pind     = Array(fstate.model.pind)
    qind     = Array(fstate.model.qind)
    sigmaind = Array(fstate.model.sigmaind)
    tauind   = Array(fstate.model.tauind)

    pE     = show_EPSP_trace(times, epsps)
    pN     = show_histogram(Nrng, Nind,
                xlabel = L"N [-]", ylabel = L"p(N)")
    truemodel != nothing && plot!([truemodel.N[1]], seriestype="vline",legend=false)
    pp     = show_histogram(prng, pind,
                xlabel = L"p [-]", ylabel = L"p(p)")
    truemodel != nothing && plot!([truemodel.p[1]], seriestype="vline",legend=false)
    pq     = show_histogram(qrng, qind,
                xlabel = L"q [A]", ylabel = L"p(q)")
    truemodel != nothing && plot!([truemodel.q[1]], seriestype="vline",legend=false)
    psigma = show_histogram(sigmarng, sigmaind,
                xlabel = L"\sigma [A]", ylabel = L"p(\sigma)")
    truemodel != nothing && plot!([truemodel.sigma[1]], seriestype="vline",legend=false)
    ptau   = show_histogram(taurng, tauind,
                xlabel = L"\tau [s]", ylabel = L"p(\tau)")
    truemodel != nothing && plot!([truemodel.tau[1]], seriestype="vline",legend=false)
    if showstates
        pn     = histogram(flatten(Array(fstate.state.n)), bins=1:20,
                    normalize = :probability,
                    xlabel = L"n", ylabel = L"p(n)")
        truestate != nothing && plot!([truestate.n[1,1]], seriestype="vline",legend=false)
        pk     = histogram(flatten(Array(fstate.state.k)), bins=1:20,
                    normalize = :probability,
                    xlabel = L"k", ylabel = L"p(k)")
        truestate != nothing && plot!([truestate.k[1,1]], seriestype="vline",legend=false)
        display(plot(pE, pN, pp, pq, psigma, ptau, pn, pk, layout = (4, 2)))
        return
    end
    display(plot(pE, pN, pp, pq, psigma, ptau, layout = (3, 2)))
end
