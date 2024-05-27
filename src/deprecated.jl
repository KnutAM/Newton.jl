function newtonsolve(x::Real, rf; kwargs...)
    @warn "Deprecated, use style with `rf(x)` as first argument" maxlog=1
    return newtonsolve(rf, x; kwargs...)
end

function newtonsolve(x0::AbstractVector, rf!, args...; kwargs...)
    @warn "Deprecated, use style with `rf(x)` as first argument" maxlog=1
    return newtonsolve(rf!, x0, args...; kwargs...)
end
