struct NewtonCache{T, LS, Tres, Tcfg}
    x::Vector{T}
    result::Tres
    linsolver::LS
    config::Tcfg
    lupivot::Vector{Int}
    blas_work::Vector{T}
end

"""
    function NewtonCache(x::AbstractVector; [linsolver])
    
Create the cache used by the `newtonsolve` and `linsolve!`. 
Only a copy of `x` will be used.

A special `linsolver` can optionally be given, please see [Linear solvers](@ref AbstractLinsolver) for more information.
"""
function NewtonCache(x::AbstractVector, rf! = Val(:newton_autotag_fun); linsolver = default_linsolver(x))
    result = DiffResults.JacobianResult(x)
    cfg = ForwardDiff.JacobianConfig(rf!, x, result.value, ForwardDiff.Chunk(length(x)))
    lupivot = Vector{Int}(undef, length(x))
    return NewtonCache(copy(x), result, linsolver, cfg, lupivot, zeros(eltype(x), 0))
end

"""
    getx(cache::NewtonCache)

Extract out the unknown values. This can be used to avoid 
allocations when solving defining the initial guess. 
"""
getx(cache::NewtonCache) = cache.x