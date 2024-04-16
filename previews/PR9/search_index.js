var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"CurrentModule = Newton","category":"page"},{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/#Standard-usage","page":"API","title":"Standard usage","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"newtonsolve\nNewtonCache\ngetx\nNewton.logging_mode","category":"page"},{"location":"api/#Newton.newtonsolve","page":"API","title":"Newton.newtonsolve","text":"newtonsolve(rf!, x0::AbstractVector, [cache::NewtonCache]; tol=1.e-6, maxiter=100)\n\nSolve the nonlinear equation (system) r(x)=0 using the newton-raphson method by calling the mutating residual function rf!(r, x), with signature rf!(r::T, x::T)::T where T<:AbstractVector x0 is the initial guess and the optional cache can be preallocated by calling NewtonCache(x0,rf!). Note that x0 is not modified, unless aliased to getx(cache).  tol is the tolerance for norm(r) and maxiter the maximum number of iterations. \n\nreturns x, drdx, converged::Bool\n\ndrdx is the derivative of r wrt. x at the returned x.\n\n\n\n\n\nnewtonsolve(rf, x0::Union{SVector,Number}; tol=1.e-6, maxiter=100)\n\nSolve the nonlinear equation (system) r(x)=0 using the newton-raphson method by calling the residual function r=rf(x), with signature rf(x::T)::T where T<:Union{SVector,Number}. x0 is the initial guess, tol the tolerance form norm(r), and maxiter the maximum number  of iterations. \n\nreturns: x, drdx, converged::Bool\n\ndrdx is the derivative of r wrt. x at the returned x.\n\n\n\n\n\n","category":"function"},{"location":"api/#Newton.NewtonCache","page":"API","title":"Newton.NewtonCache","text":"function NewtonCache(x::AbstractVector)\n\nCreate the cache used by the newtonsolve and linsolve!.  Only a copy of x will be used. \n\n\n\n\n\n","category":"type"},{"location":"api/#Newton.getx","page":"API","title":"Newton.getx","text":"getx(cache::NewtonCache)\n\nExtract out the unknown values. This can be used to avoid  allocations when solving defining the initial guess. \n\n\n\n\n\n","category":"function"},{"location":"api/#Newton.logging_mode","page":"API","title":"Newton.logging_mode","text":"Newton.logging_mode(; enable=true)\n\nHelper to turn on (enable=true) or off (enable=false) logging of iterations in Newton.jl. Internally, changes the how Newton.@if_logging expr is evaluated:  when logging mode is enabled, expr is evaluated, otherwise expr is ignored.\n\n\n\n\n\n","category":"function"},{"location":"api/#Fast-inverse","page":"API","title":"Fast inverse","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Newton.inv!","category":"page"},{"location":"api/#Newton.inv!","page":"API","title":"Newton.inv!","text":"Newton.inv!(A::Matrix, cache::NewtonCache)\n\nUtilize the LU decomposition from RecursiveFactorization.jl along with  the non-exported LinearAlgebra.inv!(::LU) to calculate the inverse of  A more efficient than inv(A). Note that A will be used as workspace and values should not be used after calling Newton.inv!\n\n\n\n\n\n","category":"function"},{"location":"api/#Use-inside-AD-calls","page":"API","title":"Use inside AD-calls","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"ad_newtonsolve","category":"page"},{"location":"api/#Newton.ad_newtonsolve","page":"API","title":"Newton.ad_newtonsolve","text":"ad_newtonsolve(rf, x0, rf_dual_args::Tuple; kwargs...)\n\nSolve rf(x, y₁(z), y₂(z), ...) = 0 to find x(y₁(z), y₂(z), ...), given the initial guess x0 as a non-dual number, and the dual input numbers  rf_dual_args = (y₁(z), y₂(z), ...) where all are derivatives wrt. the same variable z. Return x of Dual type seeded such that it corresponds to the derivative dx/dz = ∂x/∂yᵢ ⋆ dyᵢ/dz where ⋆ is the appropriate contraction.\n\nImplementation: \n\nUses the adjoint, i.e. dr/dyᵢ = 0 = ∂r/∂x ⋆ ∂x/∂yᵢ + ∂r/∂yᵢ ⇒ ∂x/∂yᵢ = -[∂r/∂x]⁻¹ ⋆ ∂r/∂yᵢ,  such that we avoid doing newton iterations with dual numbers. \n\nad_newtonsolve(rf, x0, rf_args::Tuple; kwargs...)\n\nIf rf_args do not contain dual numbers, the standard newtonsolver is just called on  f(x) = rf(x, y₁, y₂, ...), and the solution x is returned. This allows writing generic  code where the gradient is sometimes required, but not always. \n\nExample\n\nusing Newton, Tensors, ForwardDiff, BenchmarkTools\nrf(x::Vec, a::Number) = a * x - (x ⋅ x) * x\nfunction myfun!(outputs::Vector, inputs::Vector)\n    x0 = ones(Vec{2}) # Initial guess\n    a = inputs[1] + 2 * inputs[2]\n    x, converged = ad_newtonsolve(rf, x0, (a,))\n    outputs[1] = x ⋅ x\n    outputs[2] = a * x[1]\n    return outputs \nend\nout = zeros(2); inp = [1.2, 0.5]\nForwardDiff.jacobian(myfun!, out, inp)\n\ngives\n\n2×2 Matrix{Float64}:\n 1.0      2.0\n 1.57321  3.14643\n\nnote: Note\nThe maximum length of rf_dual_args where it is highly efficient is currently 5. For longer length there will be a dynamic dispatch, but this number can be extended  by adding more methods to the internal Newton.get_dual_results function.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API","title":"API","text":"This approach is faster then naively differentiating a call which includes a newtonsolve, as we avoid iterating using Dual numbers. ","category":"page"},{"location":"api/","page":"API","title":"API","text":"using Newton, Tensors, ForwardDiff, BenchmarkTools\nrf(x::Vec, a::Number) = a * x - (x ⋅ x) * x\nfunction myfun!(outputs::Vector, inputs::Vector)\n    x0 = ones(Vec{2}) # Initial guess\n    a = inputs[1] + 2 * inputs[2]\n    x, converged = ad_newtonsolve(rf, x0, (a,))\n    outputs[1] = x ⋅ x\n    outputs[2] = a * x[1]\n    return outputs \nend\nfunction myfun2!(outputs::Vector, inputs::Vector)\n    x0 = ones(Vec{2}) # Initial guess\n    a = inputs[1] + 2 * inputs[2]\n    x, _, converged = newtonsolve(x -> rf(x, a), x0)\n    outputs[1] = x ⋅ x\n    outputs[2] = a * x[1]\n    return outputs\nend\nJ = zeros(2,2)\nout = zeros(2); inp = [1.2, 0.5]\ncfg = ForwardDiff.JacobianConfig(myfun!, out, inp)\ncfg2 = ForwardDiff.JacobianConfig(myfun2!, out, inp)\n@btime ForwardDiff.jacobian!($J, $myfun2!, $out, $inp, $cfg2);  # 285.662 ns (0 allocations: 0 bytes)\n@btime myfun2!($out, $inp);                                     # 143.381 ns (0 allocations: 0 bytes)\n@btime ForwardDiff.jacobian!($J, $myfun!, $out, $inp, $cfg);    # 183.359 ns (0 allocations: 0 bytes)\n@btime myfun!($out, $inp);                                      # 143.381 ns (0 allocations: 0 bytes)","category":"page"},{"location":"api/","page":"API","title":"API","text":"showing that we get quite close to a regular non-differentiating call wrt. computational time in this microbenchmark.","category":"page"},{"location":"api/#Internal-API","page":"API","title":"Internal API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Newton.linsolve!","category":"page"},{"location":"api/#Newton.linsolve!","page":"API","title":"Newton.linsolve!","text":"linsolve!(K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)\n\nSolves the linear equation system Kx=b, mutating both K and b. b is mutated to the solution x\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Newton","category":"page"},{"location":"#Newton","page":"Home","title":"Newton","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Newton.jl provides a fast and efficient newton-raphson  solver that is suitable to be used inside a preformance critical loop. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"ForwardDiff is used for the differentiation.\nRecursiveFactorization is used for LU-factorization of regular matrices\nStaticArrays.jl and Tensors.jl are also supported","category":"page"},{"location":"","page":"Home","title":"Home","text":"A logging mode can be enabled, see Newton.logging_mode.  When more fine-grained controlled, different algorithms etc. is desired,  consider NonlinearSolve.jl. ","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(url=\"https://github.com/KnutAM/Newton.jl\")\nusing Newton","category":"page"},{"location":"#Typical-usage","page":"Home","title":"Typical usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Solve r(x)=0 by calling","category":"page"},{"location":"","page":"Home","title":"Home","text":"x, drdx, converged = newtonsolve(x::Vector, rf!::Function, cache)\nx, drdx, converged = newtonsolve(x::Union{Real,SVector}, rf::Function)","category":"page"},{"location":"#Mutating-(standard)-Array","page":"Home","title":"Mutating (standard) Array","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Initial setup (before running simulation):  Define a mutating residual function rf! which depends on  parameters, e.g. a and b, only available during the simulation.","category":"page"},{"location":"","page":"Home","title":"Home","text":"function rf!(r::Vector, x::Vector, a, b)\n    return map!(v->(exp(a*v)-b*v^2), r, x)\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"Define the unknown array x and a residual function with the signature rf!(r,x) with inputs a and b of the same type as will be used later. Then preallocate cache","category":"page"},{"location":"","page":"Home","title":"Home","text":"x=zeros(5)\na = 1.0; b=1.0\nmock_rf!(r_, x_) = rf!(r_, x_, a, b)\ncache = NewtonCache(x,mock_rf!)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Runtime setup (inside simulation): At the place where we want to solve the problem r(x)=0","category":"page"},{"location":"","page":"Home","title":"Home","text":"a, b = rand(2); # However they are calculated during simulations\ntrue_rf!(r_, x_) = rf!(r_, x_, a, b)\nx0 = getx(cache)\n# Modify x0 as you wish to provide initial guess\nx, drdx, converged = newtonsolve(x0, true_rf!, cache)","category":"page"},{"location":"","page":"Home","title":"Home","text":"It is not necessary to get x0 from the cache, but this avoids allocating it. However, this implies that x0 will be aliased to the output, i.e. x0===x after solving. ","category":"page"},{"location":"#Non-mutating-StaticArray","page":"Home","title":"Non-mutating StaticArray","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Initial setup (before running simulation):  When using static arrays, the residual function should be non-mutating, i.e. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"function rf(x::SVector, a, b)\n    return exp.(a*x) - b*x.^2\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"Runtime setup (inside simulation): At the place where we want to solve the problem r(x)=0 No cache setup is required for static arrays. Hence, get the inputs a and b, define the true residual function with signature r=rf(x), define an initial guess x0, and call the newtonsolve","category":"page"},{"location":"","page":"Home","title":"Home","text":"a=rand(); b=rand();\nrf_true(x_) = rf(x_, a, b)\nx0 = zero(SVector{5})\nx, drdx, converged = newtonsolve(x0, rf_true);","category":"page"},{"location":"","page":"Home","title":"Home","text":"which as in the mutatable array case returns a the solution vector, the jacobian at the solution and a boolean whether  the solver converged or not. ","category":"page"},{"location":"#Benchmarks","page":"Home","title":"Benchmarks","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"See benchmarks/benchmark.jl, on my laptop the results are","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> activate benchmarks/\njulia> include(\"benchmarks/benchmarks.jl\");\nBenchmark with dim=5\nrf (static):           33.099 ns (0 allocations: 0 bytes)\nrf (dynamic):          32.931 ns (0 allocations: 0 bytes)\nnewtonsolve static:    1.000 μs (0 allocations: 0 bytes)\nnewtonsolve dynamic:   2.400 μs (11 allocations: 1.50 KiB)\nnlsolve dynamic:       6.900 μs (58 allocations: 6.23 KiB)\n\nBenchmark with dim=10\nrf (static):           61.491 ns (0 allocations: 0 bytes)\nrf (dynamic):          66.187 ns (0 allocations: 0 bytes)\nnewtonsolve static:    4.200 μs (0 allocations: 0 bytes)\nnewtonsolve dynamic:   5.100 μs (7 allocations: 5.28 KiB)\nnlsolve dynamic:       11.400 μs (58 allocations: 12.25 KiB)\n\nBenchmark with dim=20\nrf (static):           119.333 ns (0 allocations: 0 bytes)\nrf (dynamic):          125.471 ns (0 allocations: 0 bytes)\nnewtonsolve static:    7.900 μs (16 allocations: 14.81 KiB)\nnewtonsolve dynamic:   14.600 μs (5 allocations: 4.38 KiB)\nnlsolve dynamic:       29.100 μs (62 allocations: 23.39 KiB)\n\nBenchmark with dim=40\nrf (static):           265.634 ns (0 allocations: 0 bytes)\nrf (dynamic):          251.370 ns (0 allocations: 0 bytes)\nnewtonsolve static:    38.600 μs (16 allocations: 53.69 KiB)\nnewtonsolve dynamic:   53.200 μs (5 allocations: 4.38 KiB)\nnlsolve dynamic:       83.400 μs (67 allocations: 55.67 KiB)","category":"page"},{"location":"","page":"Home","title":"Home","text":"showing that static arrays are faster than dynamic arrays with newtonsolve and that newtonsolve outperforms nlsolve in these specific cases. (nlsolve does not  support StaticArrays.)","category":"page"}]
}
