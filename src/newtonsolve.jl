"""
    newtonsolve(rf!, x0::AbstractVector, [cache::NewtonCache]; tol=1.e-6, maxiter=100)

Solve the nonlinear equation (system) `r(x)=0` using the newton-raphson method by calling
the mutating residual function `rf!(r, x)`, with signature `rf!(r::T, x::T)::T where T<:AbstractVector`
`x0` is the initial guess and the optional `cache` can be preallocated by calling `NewtonCache(x0,rf!)`.
Note that `x0` is not modified, unless aliased to `getx(cache)`. 
`tol` is the tolerance for `norm(r)` and `maxiter` the maximum number of iterations. 

returns `x, drdx, converged::Bool`

`drdx` is the derivative of r wrt. x at the returned `x`.
"""
function newtonsolve(rf!::F, x0::AbstractVector, cache::NewtonCache = NewtonCache(x0,rf!); tol=1.e-6, maxiter=100) where F
    diffresult = cache.result
    x = getx(cache)
    copyto!(x, x0)
    cfg = cache.config
    drdx = DiffResults.jacobian(diffresult)
    r = DiffResults.value(diffresult)
    @if_logging errs = zeros(maxiter)
    @if_logging resids = Vector{Float64}[]
    for i = 1:maxiter
        # Disable checktag using Val{false}(). solve_residual should never be differentiated using dual numbers! 
        # This is required when using a different (but equivalent) anynomus function for caching than for running.
        # Note that this shows up as dynamic dispatch within chunk_mode_jacobian, but doesn't affect allocations/performance.
        ForwardDiff.jacobian!(diffresult, rf!, r, x, cfg, Val(false))
        err = norm(r)

        @if_logging errs[i] = err
        @if_logging push!(resids, copy(r))

        check_no_dual(err) # Check that we don't try to differentiate (gets compiled away for type-stable code)
        if err < tol
            return x, drdx, true
        end
        minus_dx = linsolve!(drdx, r, cache)    # minus_dx aliases r
        x .-= minus_dx
    end
    # No convergence
    @if_logging show_iteration_trace(errs, resids, tol)
    return x, drdx, false
end

@inline check_no_dual(::Number) = nothing
@inline check_no_dual(::ForwardDiff.Dual) = throw(ArgumentError("newtonsolve cannot be differentiated"))

"""
    newtonsolve(rf, x0::T; tol=1.e-6, maxiter=100) where {T <: 
        Union{Number, StaticArrays.SVector, Tensors.Vec, Tensors.SecondOrderTensor}}
    
Solve the nonlinear equation (system) `r(x)=0` using the newton-raphson method by calling
the residual function `r=rf(x)`, with signature `rf(x::T)::T`
`x0::T` is the initial guess, `tol` the tolerance form `norm(r)`, and `maxiter` the maximum number 
of iterations. 

returns: `x, drdx, converged::Bool`

`drdx` is the derivative of r wrt. x at the returned `x`.
"""
function newtonsolve(rf::F, x::T; tol=1.e-6, maxiter=100) where {F, T <: Union{Number, SVector, Vec, SecondOrderTensor}}
    local drdx
    @if_logging errs = zeros(maxiter)
    @if_logging resids = zeros(T, maxiter)
    for i = 1:maxiter
        r = rf(x)
        err = norm(r)
        @if_logging errs[i] = err
        @if_logging resids[i] = r
        drdx = generic_jacobian(rf, x)
        if err < tol
            return x, drdx, true
        end
        x -= linsolve(drdx, r)
    end
    @if_logging show_iteration_trace(errs, resids, tol)
    return x, drdx, false
end

generic_jacobian(rf::F, x::Number) where F = ForwardDiff.derivative(rf, x)
generic_jacobian(rf::F, x::SVector) where F = ForwardDiff.jacobian(rf, x)
generic_jacobian(rf::F, x::Union{Tensors.Vec, Tensors.SecondOrderTensor}) where F = Tensors.gradient(rf, x)