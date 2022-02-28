var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = Newton","category":"page"},{"location":"#Newton","page":"Home","title":"Newton","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The goal of the small Newton.jl package is to provide a fast and efficient newton-raphson solver for nonlinear equation systems, suitable to be used inside a preformance critical loop. It is mostly tested for small equations systems (<100 variables). When more fine-grained controlled over algorithms or more iteration information is desired, using NLsolve is recommended.","category":"page"},{"location":"#Basic-usage","page":"Home","title":"Basic usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using Newton","category":"page"},{"location":"","page":"Home","title":"Home","text":"Define a mutating residual function rf!","category":"page"},{"location":"","page":"Home","title":"Home","text":"nsize=4\n(a,b) = [rand(nsize) for _ in 1:2]\nfunction rf!(r, x)\n    r .= - a + b.*x + exp.(x)\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"Define initial guess x","category":"page"},{"location":"","page":"Home","title":"Home","text":"x=zeros(nsize)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Preallocate cache ","category":"page"},{"location":"","page":"Home","title":"Home","text":"cache = NewtonCache(x,rf!)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Solve the problem r=0","category":"page"},{"location":"","page":"Home","title":"Home","text":"drdx = get_drdx(cache)  # Alternatively drdx=zeros(nsize,nsize), but this allocates \nconverged = newtonsolve!(x, drdx, rf!, cache)","category":"page"},{"location":"#Speed-and-allocation-comparison","page":"Home","title":"Speed and allocation comparison","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"See benchmarks/benchmark.jl, on my laptop the results are","category":"page"},{"location":"","page":"Home","title":"Home","text":"include(\"benchmarks/benchmark.jl\")\n@btime rf!\n  200.000 ns (0 allocations: 0 bytes)\n@btime newtonsolve!\n  29.400 μs (7 allocations: 6.12 KiB)\n@btime nlsolve\n  51.500 μs (74 allocations: 40.98 KiB)\nBenchmark (dim=20) complete","category":"page"},{"location":"","page":"Home","title":"Home","text":"showing that newtonsolve! is approximately 1.75 times faster than the basic usage of nlsolve for this particular case.","category":"page"},{"location":"#Exported-API","page":"Home","title":"Exported API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [Newton]\nPrivate = false","category":"page"},{"location":"#Newton.NewtonCache-Tuple{AbstractVector, Any}","page":"Home","title":"Newton.NewtonCache","text":"function NewtonCache(x::AbstractVector, rf!)\n\nCreate the cache used by the newtonsolve! and linsolve!  to find x such that rf!(r,x) yields r=0.\n\n\n\n\n\n","category":"method"},{"location":"#Newton.get_drdx-Tuple{NewtonCache}","page":"Home","title":"Newton.get_drdx","text":"get_drdx(cache::NewtonCache) = DiffResults.jacobian(cache.result)\n\n\n\n\n\n","category":"method"},{"location":"#Newton.newtonsolve!","page":"Home","title":"Newton.newtonsolve!","text":"newtonsolve!(x::AbstractVector, drdx::AbstractMatrix, rf!, cache::ResidualCache; tol=1.e-6, max_iter=100)\n\nSolve the nonlinear equation system r(x)=0 using the newton-raphson method. Returns true if converged and false otherwise.\n\nargs\n\nx: Vector of unknowns. Provide as initial guess, mutated to solution.\ndrdx: Jacobian matrix. Only provided as preallocation. Can be aliased to DiffResults.jacobian(cache.result)\nrf!: Residual function. Signature rf!(r, x) and mutating the residual r\ncache: Optional cache that can be preallocated by calling ResidualCache(x, rf!)\n\nkwargs\n\ntol=1.e-6: Tolerance on norm(r)\nmaxiter=100: Maximum number of iterations before no convergence\n\n\n\n\n\n","category":"function"},{"location":"#Internal-API","page":"Home","title":"Internal API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [Newton]\nPublic = false","category":"page"},{"location":"#Newton.linsolve!-Tuple{AbstractMatrix, AbstractVector, NewtonCache}","page":"Home","title":"Newton.linsolve!","text":"linsolve!(K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)\n\nSolves the linear equation system Kx=b, mutating both K and b. b is mutated to the solution x\n\n\n\n\n\n","category":"method"}]
}