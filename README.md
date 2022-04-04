# Newton.jl

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://KnutAM.github.io/Newton.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://KnutAM.github.io/Newton.jl/dev)
[![Build Status](https://github.com/KnutAM/Newton.jl/workflows/CI/badge.svg)](https://github.com/KnutAM/Newton.jl/actions)
[![Coverage](https://codecov.io/gh/KnutAM/Newton.jl/branch/main/graph/badge.svg?token=9JRHlQ6meT)](https://codecov.io/gh/KnutAM/Newton.jl)

Newton.jl provides an efficient newton-raphson solver for nonlinear equation systems. The main goal is to keep the allocations low and the speed high (duh).

Its purpose is to be used inside a preformance critical loop, and is mostly tested for small equations systems (<100 variables). When more fine-grained controlled over algorithms or more iteration information are desired, [NLsolve](https://github.com/JuliaNLSolvers/NLsolve.jl) has many more options.

## Installation
```julia
Pkg.add(url="https://github.com/KnutAM/Newton.jl")
```
