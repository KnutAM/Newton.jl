module Newton
using LinearAlgebra
using RecursiveFactorization
using DiffResults
using ForwardDiff
using StaticArrays

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
    newtonsolve!(x::AbstractVector, drdx::AbstractMatrix, rf!, cache::ResidualCache; tol=1.e-6, maxiter=100)

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
function newtonsolve!(x::AbstractVector, drdx::AbstractMatrix, rf!, cache::NewtonCache = NewtonCache(x,rf!); tol=1.e-6, maxiter=100)
    diffresult = cache.result
    cfg = cache.config
    for i = 1:maxiter
        # Disable checktag using Val{false}(). solve_residual should never be differentiated using dual numbers! 
        # This is required when using a different (but equivalent) anynomus function for caching than for running.
        ForwardDiff.jacobian!(diffresult, rf!, diffresult.value, x, cfg, Val{false}())
        err = norm(DiffResults.value(diffresult))
        i == 1 && @assert !(typeof(err) <: ForwardDiff.Dual)    # Check that we don't try to differentiate
        if err < tol
            drdx .= DiffResults.jacobian(diffresult)
            return true
        end
        linsolve!(DiffResults.jacobian(diffresult), DiffResults.value(diffresult), cache)
        x .-= DiffResults.value(diffresult)
    end
    # No convergence
    return false
end

"""
    linsolve(drdx::SMatrix{dim,dim}, r::SVector{dim}) where{dim}

Solves the linear equation system `drdx*x=r` without mutating and returns the solution x
"""
@inline linsolve(drdx::SMatrix{dim,dim}, r::SVector{dim}) where{dim} = drdx\r

"""
    newtonsolve(x::SVector, rf; tol=1.e-6, max_iter=100)

Solve the nonlinear equation system `r(x)=0` using the newton-raphson method.
Returns type: `(converged, x, drdx)`, SVector, SMatrix)` where 
- `converged::Bool` is `true`` if converged and `false` otherwise
- `x::SVector` is the solution vector such that `r(x)=0`
- `drdx::SMatrix` is the jacobian at `x`

# args
- `x`: Vector of unknowns. Provide as initial guess, mutated to solution.
- `rf`: Residual function. Signature `r=rf(x::SVector{dim})::SVector{dim}`

# kwargs
- `tol=1.e-6`: Tolerance on `norm(r)`
- `maxiter=100`: Maximum number of iterations before no convergence

"""
function newtonsolve(x::SVector{dim}, rf; tol=1.e-6, maxiter=100) where{dim}
    for _ = 1:maxiter
        r = rf(x)
        err = norm(r)
        drdx = ForwardDiff.jacobian(rf, x)
        if err < tol
            return true, x, drdx
        end
        x -= linsolve(drdx, r)
    end
    return false, zero(SVector{dim})*NaN, zero(SMatrix{dim,dim})*NaN
end
    
export newtonsolve!, newtonsolve
export NewtonCache
export get_drdx

end
