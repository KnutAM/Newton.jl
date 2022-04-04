```@meta
CurrentModule = Newton
```

# Newton
The goal of the small [Newton.jl](https://github.com/KnutAM/Newton.jl) package is to provide a fast and efficient newton-raphson solver for nonlinear equation systems, suitable to be used inside a preformance critical loop. It is mostly tested for small equations systems (<100 variables). When more fine-grained controlled over algorithms or more iteration information is desired, using [NLsolve](https://github.com/JuliaNLSolvers/NLsolve.jl) is recommended.

## Basic usage
### Mutating standard array
```julia
using Newton
```

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

### Speed and allocation comparison
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


## Using StaticArrays
```julia
using Newton
```

Define a non-mutating residual function
```julia
function rf(x::SVector)
    return exp.(x) - x.^2
end
```

Provide an initial guess
```julia
x_s = zero(SVector{dim})
```

Find the `x` that solves the non-linear equation system `r(x)=0`, as well as the jacobian `drdx` at that `x`,
```julia
converged, x, drdx = newtonsolve($x_s, $rf);
```

### Speed comparison
See `benchmarks/benchmark_static.jl`, on my laptop the results are
```julia
include("benchmarks/benchmark_static.jl")
Benchmark with dim=5
rf (static):           18.637 ns (0 allocations: 0 bytes)
rf (dynamic):          18.136 ns (0 allocations: 0 bytes)
newtonsolve static:    600.000 ns (0 allocations: 0 bytes)
newtonsolve dynamic:   1.500 μs (9 allocations: 1.22 KiB)
nlsolve dynamic:       4.300 μs (58 allocations: 6.23 KiB)

Benchmark with dim=10
rf (static):           35.247 ns (0 allocations: 0 bytes)
rf (dynamic):          34.844 ns (0 allocations: 0 bytes)
newtonsolve static:    2.600 μs (0 allocations: 0 bytes)
newtonsolve dynamic:   3.100 μs (5 allocations: 4.38 KiB)
nlsolve dynamic:       6.500 μs (58 allocations: 12.25 KiB)

Benchmark with dim=20
rf (static):           68.098 ns (0 allocations: 0 bytes)
rf (dynamic):          68.275 ns (0 allocations: 0 bytes)
newtonsolve static:    5.200 μs (16 allocations: 14.81 KiB)
newtonsolve dynamic:   8.400 μs (5 allocations: 4.38 KiB)
nlsolve dynamic:       16.400 μs (62 allocations: 23.39 KiB)
```
showing that using StaticArrays will be significantly faster with `newtonsolve`. (`nlsolve` does not  support StaticArrays.)

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
