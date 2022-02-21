using Documenter, BinomialSynapses

makedocs(
    sitename="BinomialSynapses.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [BinomialSynapses]
)

deploydocs(
    repo = "github.com/Theoretical-Neuroscience-Group/BinomialSynapses.jl.git",
)
