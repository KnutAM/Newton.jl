```@meta
CurrentModule = Newton
```

# Newton
The goal of the small [Newton.jl](https://github.com/KnutAM/Newton.jl) package is to provide a fast and efficient newton-raphson solver for nonlinear equation systems, suitable to be used inside an expensive loop. It is mostly tested for small equations systems (<100 variables). When more fine-grained controlled over algorithms or more iteration information is desired, using [NLsolve](https://github.com/JuliaNLSolvers/NLsolve.jl) is recommended.

## Basic usage
Define a mutating residual function `rf!`
```julia
nsize=4
(a,b) = [rand(nsize) for _ in 1:2]
function rf!(r, x)
    r .= - a + b.*x + exp.(x)
end
```

Define initial guess `x`
```julia
x=zeros(nsize)
```
Preallocate `cache` 
```julia
cache = NewtonCache(x,rf!)
```

Solve the problem `r=0`
```julia
drdx = get_drdx(cache)  # Alternatively drdx=zeros(nsize,nsize), but this allocates 
converged = newtonsolve!(x, drdx, rf!, cache)
```

## Speed and allocation comparison
See `benchmarks/benchmark.jl`, on my laptop the results are
```julia
include("benchmarks/benchmark.jl")
@btime rf!
  200.000 ns (0 allocations: 0 bytes)
@btime newtonsolve!
  29.400 μs (7 allocations: 6.12 KiB)
@btime nlsolve
  51.500 μs (74 allocations: 40.98 KiB)
Benchmark (dim=20) complete
```
showing that `newtonsolve!` is approximately 1.75 times faster than the basic usage of `nlsolve` for this particular case.

## Exported API
```@autodocs
Modules = [Newton]
Private = false
```

## Internal API
```@autodocs
Modules = [Newton]
Public = false
```