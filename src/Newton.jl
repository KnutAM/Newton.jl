module Newton
using LinearAlgebra
using RecursiveFactorization
using DiffResults
using ForwardDiff
using StaticArrays

struct NewtonCache{T,Tres,Tcfg}
    x::Vector{T}
    result::Tres
    config::Tcfg
    lupivot::Vector{Int}
end

"""
    function NewtonCache(x::AbstractVector, rf!)
    
Create the cache used by the `newtonsolve` and `linsolve!`. 
Only a copy of `x` will be used. 
"""
function NewtonCache(x::AbstractVector, rf!)
    result = DiffResults.JacobianResult(x)
    cfg = ForwardDiff.JacobianConfig(rf!, x, result.value, ForwardDiff.Chunk(length(x)))
    lupivot = Vector{Int}(undef, length(x))
    return NewtonCache(copy(x), result, cfg, lupivot)
end

"""
    getx(cache::NewtonCache)

Extract out the unknown values. This can be used to avoid 
allocations when solving defining the initial guess. 
"""
getx(cache::NewtonCache) = cache.x

"""
    linsolve!(K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)

Solves the linear equation system `Kx=b`, mutating both `K` and `b`.
`b` is mutated to the solution `x`
"""
function linsolve!(K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)
    LU = RecursiveFactorization.lu!(K, cache.lupivot, Val{true}(), Val{false}())
    ldiv!(LU, b)
    return b
end

"""
    newtonsolve(x0::AbstractVector, rf!, [cache::NewtonCache]; tol=1.e-6, maxiter=100)

Solve the nonlinear equation (system) `r(x)=0` using the newton-raphson method by calling
the mutating residual function `rf!(r, x)`, with signature `rf!(r::T, x::T)::T where T<:AbstractVector`
`x0` is the initial guess and the optional `cache` can be preallocated by calling `NewtonCache(x0,rf!)`.
Note that `x0` is not modified, unless aliased to `getx(cache)`. 
`tol` is the tolerance for `norm(r)` and `maxiter` the maximum number of iterations. 

returns `x, drdx, converged::Bool`

`drdx` is the derivative of r wrt. x at the returned `x`.
"""
function newtonsolve(x0::AbstractVector, rf!, cache::NewtonCache = NewtonCache(x0,rf!); tol=1.e-6, maxiter=100)
    diffresult = cache.result
    x = getx(cache)
    copy!(x, x0)
    cfg = cache.config
    drdx = DiffResults.jacobian(diffresult)
    r = DiffResults.value(diffresult)
    for i = 1:maxiter
        # Disable checktag using Val{false}(). solve_residual should never be differentiated using dual numbers! 
        # This is required when using a different (but equivalent) anynomus function for caching than for running.
        ForwardDiff.jacobian!(diffresult, rf!, r, x, cfg, Val{false}())
        err = norm(r)
        # Check that we don't try to differentiate:
        i == 1 && check_no_dual(err)
        if err < tol
            return x, drdx, true
        end
        linsolve!(drdx, r, cache)
        x .-= r # Note: r mutated to drdx\r
    end
    # No convergence
    return x, drdx, false
end

check_no_dual(::Number) = nothing
check_no_dual(::ForwardDiff.Dual) = throw(ArgumentError("newtonsolve cannot be differentiated"))

"""
    newtonsolve(x0::Union{SVector,Number}, rf; tol=1.e-6, maxiter=100)

Solve the nonlinear equation (system) `r(x)=0` using the newton-raphson method by calling
the residual function `r=rf(x)`, with signature `rf(x::T)::T where T<:Union{SVector,Number}`.
`x0` is the initial guess, `tol` the tolerance form `norm(r)`, and `maxiter` the maximum number 
of iterations. 

returns: `x, drdx, converged::Bool`

`drdx` is the derivative of r wrt. x at the returned `x`.
"""
function newtonsolve(x::SVector{dim}, rf; tol=1.e-6, maxiter=100) where{dim}
    local drdx
    for _ = 1:maxiter
        r = rf(x)
        err = norm(r)
        drdx = ForwardDiff.jacobian(rf, x)
        if err < tol
            return x, drdx, true
        end
        x -= drdx\r
    end
    return x, drdx, false
end
    
function newtonsolve(x::Real, rf; tol=1.e-6, maxiter=100)
    local drdx
    for _ = 1:maxiter
        r = rf(x)
        err = norm(r)
        drdx = ForwardDiff.derivative(rf, x)
        if err < tol 
            return x, drdx, true
        end
        x -= r/drdx
    end
    return x, drdx, false
end

export newtonsolve
export NewtonCache
export getx

end
