struct NewtonCache{T,Tres,Tcfg}
    x::Vector{T}
    result::Tres
    config::Tcfg
    lupivot::Vector{Int}
    blas_work::Vector{T}
end

"""
    function NewtonCache(x::AbstractVector)
    
Create the cache used by the `newtonsolve` and `linsolve!`. 
Only a copy of `x` will be used. 
"""
function NewtonCache(x::AbstractVector, rf! = Val(:newton_autotag_fun))
    result = DiffResults.JacobianResult(x)
    cfg = ForwardDiff.JacobianConfig(rf!, x, result.value, ForwardDiff.Chunk(length(x)))
    lupivot = Vector{Int}(undef, length(x))
    return NewtonCache(copy(x), result, cfg, lupivot, zeros(eltype(x),0))
end

"""
    getx(cache::NewtonCache)

Extract out the unknown values. This can be used to avoid 
allocations when solving defining the initial guess. 
"""
getx(cache::NewtonCache) = cache.x