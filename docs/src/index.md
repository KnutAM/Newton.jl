```@meta
CurrentModule = Newton
```

# Newton
The goal of the small [Newton.jl](https://github.com/KnutAM/Newton.jl) package is to provide a fast and efficient newton-raphson solver for nonlinear equation systems, suitable to be used inside a preformance critical loop. A key feature is that the jacobian at the solution is returned. It is mostly tested for small equations systems (<100 variables). When more fine-grained controlled over algorithms or more iteration information is desired, using [NLsolve](https://github.com/JuliaNLSolvers/NLsolve.jl) is recommended.

## Usage
### Mutating standard array
```julia
using Newton
```

Define a mutating residual function `rf!`
```julia
function rf!(r::Vector, x::Vector)
    return map!(v->(exp(v)-v^2), r, x)
end
```

Define the unknown array `x` and preallocate cache `cache`
```julia
x=zeros(5)
cache = NewtonCache(x,rf!)
```

At the place where we want to solve the problem `r=0`
```julia
x0 = getx(cache)
# Modify x0 as you wish to provide initial guess
x, drdx, converged = newtonsolve!(x0, rf!, cache)
```
It is not necessary to get `x0` from the cache, but this avoids any allocations. This implies that `x0` will be aliased to the output, i.e. `x0===x` after solving. 


### Using StaticArrays
When using static arrays, the residual function should be non-mutating, i.e. 
```julia
function rf(x::SVector)
    return exp.(x) - x.^2
end
```

No cache setup is required for static arrays. Hence, define an initial guess `x0` and call the `newtonsolve`
```julia
x0 = zero(SVector{dim})
x, drdx, converged = newtonsolve(x0, rf);
```
which as in the mutatable array case returns a the solution
vector, the jacobian at the solution and a boolean whether 
the solver converged or not. 

### Speed comparison
See `benchmarks/benchmark.jl`, on my laptop the results are
```julia
pkg> activate benchmarks/
julia> include("benchmarks/benchmarks.jl");
Benchmark with dim=5
rf (static):           33.099 ns (0 allocations: 0 bytes)
rf (dynamic):          32.931 ns (0 allocations: 0 bytes)
newtonsolve static:    1.000 μs (0 allocations: 0 bytes)
newtonsolve dynamic:   2.400 μs (11 allocations: 1.50 KiB)
nlsolve dynamic:       6.900 μs (58 allocations: 6.23 KiB)

Benchmark with dim=10
rf (static):           61.491 ns (0 allocations: 0 bytes)
rf (dynamic):          66.187 ns (0 allocations: 0 bytes)
newtonsolve static:    4.200 μs (0 allocations: 0 bytes)
newtonsolve dynamic:   5.100 μs (7 allocations: 5.28 KiB)
nlsolve dynamic:       11.400 μs (58 allocations: 12.25 KiB)

Benchmark with dim=20
rf (static):           119.333 ns (0 allocations: 0 bytes)
rf (dynamic):          125.471 ns (0 allocations: 0 bytes)
newtonsolve static:    7.900 μs (16 allocations: 14.81 KiB)
newtonsolve dynamic:   14.600 μs (5 allocations: 4.38 KiB)
nlsolve dynamic:       29.100 μs (62 allocations: 23.39 KiB)

Benchmark with dim=40
rf (static):           265.634 ns (0 allocations: 0 bytes)
rf (dynamic):          251.370 ns (0 allocations: 0 bytes)
newtonsolve static:    38.600 μs (16 allocations: 53.69 KiB)
newtonsolve dynamic:   53.200 μs (5 allocations: 4.38 KiB)
nlsolve dynamic:       83.400 μs (67 allocations: 55.67 KiB)
```
showing that static arrays are faster than dynamic arrays with `newtonsolve` and that `newtonsolve` outperforms `nlsolve` in these specific cases. (`nlsolve` does not  support StaticArrays.)

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
