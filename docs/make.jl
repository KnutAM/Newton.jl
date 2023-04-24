using Newton
using Documenter

DocMeta.setdocmeta!(Newton, :DocTestSetup, :(using Newton); recursive=true)

makedocs(;
    modules=[Newton],
    authors="Knut Andreas Meyer and contributors",
    repo="https://github.com/KnutAM/Newton.jl/blob/{commit}{path}#{line}",
    sitename="Newton.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://KnutAM.github.io/Newton.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    devbranch="main",
    repo="github.com/KnutAM/Newton.jl",
    push_preview=true,
)
