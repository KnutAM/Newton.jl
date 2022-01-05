module Newton
using LinearAlgebra
using RecursiveFactorization
using DiffResults
using ForwardDiff

struct NewtonCache{Tres,Tcfg}
    result::Tres
    config::Tcfg
    lupivot::Vector{Int}
end

"""
    function NewtonCache(x::AbstractVector, rf!)
    
    Create the cache used by the `newtonsolve!` and `linsolve!` 
    to find `x` such that `rf!(r,x)` yields `r=0`.
"""
function NewtonCache(x::AbstractVector, rf!)
    result = DiffResults.JacobianResult(x)
    cfg = ForwardDiff.JacobianConfig(rf!, x, result.value, ForwardDiff.Chunk(length(x)))
    lupivot = Vector{Int}(undef, length(x))
    return NewtonCache(result, cfg, lupivot)
end

"""
    get_drdx(cache::NewtonCache) = DiffResults.jacobian(cache.result)    

"""
get_drdx(cache::NewtonCache) = DiffResults.jacobian(cache.result)

"""
    linsolve!(K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)

Solves the linear equation system `Kx=b`, mutating both `K` and `b`.
`b` is mutated to the solution `x`
"""
function linsolve!(K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)
    LU = RecursiveFactorization.lu!(K, cache.lupivot)
    ldiv!(LU, b)
    return b
end

"""
    newtonsolve!(x::AbstractVector, drdx::AbstractMatrix, rf!, cache::ResidualCache; tol=1.e-6, max_iter=100)

Solve the nonlinear equation system r(x)=0 using the newton-raphson method. Returns `true` if converged and `false` otherwise.

# args
- `x`: Vector of unknowns. Provide as initial guess, mutated to solution.
- `drdx`: Jacobian matrix. Only provided as preallocation. Can be aliased to `DiffResults.jacobian(cache.result)`
- `rf!`: Residual function. Signature `rf!(r, x)` and mutating the residual `r`
- `cache`: Optional cache that can be preallocated by calling `ResidualCache(x, rf!)`

# kwargs
- `tol=1.e-6`: Tolerance on `norm(r)`
- `maxiter=100`: Maximum number of iterations before no convergence

"""
function newtonsolve!(x::AbstractVector, drdx::AbstractMatrix, rf!, cache::NewtonCache = NewtonCache(x,rf!); tol=1.e-6, max_iter=100)
    diffresult = cache.result
    cfg = cache.config
    for _ = 1:max_iter
        # Disable checktag using Val{false}(). solve_residual should never be differentiated using dual numbers! 
        # This is required when using a different (but equivalent) anynomus function for caching than for running.
        ForwardDiff.jacobian!(diffresult, rf!, diffresult.value, x, cfg, Val{false}())
        err = norm(diffresult.value)
        if err < tol
            drdx .= DiffResults.jacobian(diffresult)
            return true
        end
        linsolve!(DiffResults.jacobian(diffresult), diffresult.value, cache)
        x .-= diffresult.value
    end
    # No convergence
    return false
end

export newtonsolve!
export NewtonCache
export get_drdx

end
