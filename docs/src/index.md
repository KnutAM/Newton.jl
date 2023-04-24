```@meta
CurrentModule = Newton
```

# Newton
[Newton.jl](https://github.com/KnutAM/Newton.jl) provides a fast and efficient newton-raphson 
solver that is suitable to be used inside a preformance critical loop. 

* [`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl) is used for the differentiation.
* [`RecursiveFactorization`](https://github.com/JuliaLinearAlgebra/RecursiveFactorization.jl) is used for LU-factorization of regular matrices
* [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl/) are also supported

A logging mode can be enabled, see [`Newton.logging_mode`](@ref). 
When more fine-grained controlled, different algorithms etc. is desired, 
consider [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/). 

## Installation
```julia
using Pkg
Pkg.add(url="https://github.com/KnutAM/Newton.jl")
using Newton
```

## Typical usage
Solve `r(x)=0` by calling

* `x, drdx, converged = newtonsolve(x::Vector, rf!::Function, cache)`
* `x, drdx, converged = newtonsolve(x::Union{Real,SVector}, rf::Function)`


### Mutating (standard) `Array`

**Initial setup** (before running simulation): 
Define a mutating residual function `rf!` which depends on 
parameters, e.g. `a` and `b`, only available during the simulation.
```julia
function rf!(r::Vector, x::Vector, a, b)
    return map!(v->(exp(a*v)-b*v^2), r, x)
end
```

Define the unknown array `x` and a residual function with the signature `rf!(r,x)` with inputs `a` and `b` of the same type as will be used later. Then preallocate `cache`
```julia
x=zeros(5)
a = 1.0; b=1.0
mock_rf!(r_, x_) = rf!(r_, x_, a, b)
cache = NewtonCache(x,mock_rf!)
```

**Runtime setup** (inside simulation): At the place where we want to solve the problem `r(x)=0`
```julia
a, b = rand(2); # However they are calculated during simulations
true_rf!(r_, x_) = rf!(r_, x_, a, b)
x0 = getx(cache)
# Modify x0 as you wish to provide initial guess
x, drdx, converged = newtonsolve(x0, true_rf!, cache)
```
It is not necessary to get `x0` from the cache, but this avoids allocating it. However, this implies that `x0` will be aliased to the output, i.e. `x0===x` after solving. 

### Non-mutating `StaticArray`
**Initial setup** (before running simulation): 
When using static arrays, the residual function should be non-mutating, i.e. 
```julia
function rf(x::SVector, a, b)
    return exp.(a*x) - b*x.^2
end
```

**Runtime setup** (inside simulation): At the place where we want to solve the problem `r(x)=0`
No cache setup is required for static arrays. Hence, get the inputs `a` and `b`, define the true residual function with signature `r=rf(x)`, define an initial guess `x0`, and call the `newtonsolve`
```julia
a=rand(); b=rand();
rf_true(x_) = rf(x_, a, b)
x0 = zero(SVector{5})
x, drdx, converged = newtonsolve(x0, rf_true);
```
which as in the mutatable array case returns a the solution
vector, the jacobian at the solution and a boolean whether 
the solver converged or not. 

## Benchmarks
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