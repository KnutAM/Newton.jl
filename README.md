# Newton.jl

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://KnutAM.github.io/Newton.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://KnutAM.github.io/Newton.jl/dev)
[![Build Status](https://github.com/KnutAM/Newton.jl/workflows/CI/badge.svg)](https://github.com/KnutAM/Newton.jl/actions)
[![Coverage](https://codecov.io/gh/KnutAM/Newton.jl/branch/main/graph/badge.svg?token=9JRHlQ6meT)](https://codecov.io/gh/KnutAM/Newton.jl)

[Newton.jl](https://github.com/KnutAM/Newton.jl) provides a fast and efficient newton-raphson 
solver that is suitable to be used inside a preformance critical loop. 

When more fine-grained controlled, different algorithms etc. is desired, 
consider [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/). 

## Installation
```julia
Pkg.add(url="https://github.com/KnutAM/Newton.jl")
```
