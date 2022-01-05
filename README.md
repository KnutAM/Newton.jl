# Newton.jl

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://KnutAM.github.io/Newton.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://KnutAM.github.io/Newton.jl/dev)
[![Build Status](https://github.com/KnutAM/Newton.jl/workflows/CI/badge.svg)](https://github.com/KnutAM/Newton.jl/actions)
[![Coverage](https://codecov.io/gh/KnutAM/Newton.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/KnutAM/Newton.jl)

Newton.jl provides a fast and efficient newton-raphson solver for nonlinear equation systems. It is suitable to be used inside a preformance critical loop. It is mostly tested for small equations systems (<100 variables). When more fine-grained controlled over algorithms or more iteration information is desired, using [NLsolve](https://github.com/JuliaNLSolvers/NLsolve.jl) is recommended.

