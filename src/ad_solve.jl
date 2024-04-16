const DualInput = Union{<:ForwardDiff.Dual, <:AbstractArray{TD} where {TD<:ForwardDiff.Dual}}
const NonDualInput = Union{<:AbstractFloat, <:AbstractArray{TD} where {TD<:AbstractFloat}}

 """
    ad_newtonsolve(rf, x0, rf_dual_args::Tuple; kwargs...)

Solve `rf(x, y₁(z), y₂(z), ...) = 0` to find `x(y₁(z), y₂(z), ...)`,
given the initial guess `x0` as a non-dual number, and the dual input numbers 
`rf_dual_args = (y₁(z), y₂(z), ...)` where all are derivatives wrt. the same variable `z`.
Return `x` of `Dual` type seeded such that it corresponds to the derivative
`dx/dz = ∂x/∂yᵢ ⋆ dyᵢ/dz` where `⋆` is the appropriate contraction.

**Implementation:** 

Uses the adjoint, i.e. `dr/dyᵢ = 0 = ∂r/∂x ⋆ ∂x/∂yᵢ + ∂r/∂yᵢ ⇒ ∂x/∂yᵢ = -[∂r/∂x]⁻¹ ⋆ ∂r/∂yᵢ`, 
such that we avoid doing newton iterations with dual numbers. 

    ad_newtonsolve(rf, x0, rf_args::Tuple; kwargs...)

If `rf_args` do not contain dual numbers, the standard newtonsolver is just called on 
`f(x) = rf(x, y₁, y₂, ...)`, and the solution `x` is returned. This allows writing generic 
code where the gradient is sometimes required, but not always. 

## Example
```julia
using Newton, Tensors, ForwardDiff, BenchmarkTools
rf(x::Vec, a::Number) = a * x - (x ⋅ x) * x
function myfun!(outputs::Vector, inputs::Vector)
    x0 = ones(Vec{2}) # Initial guess
    a = inputs[1] + 2 * inputs[2]
    x, converged = ad_newtonsolve(rf, x0, (a,))
    outputs[1] = x ⋅ x
    outputs[2] = a * x[1]
    return outputs 
end
out = zeros(2); inp = [1.2, 0.5]
ForwardDiff.jacobian(myfun!, out, inp)
```
gives
```
2×2 Matrix{Float64}:
 1.0      2.0
 1.57321  3.14643
```

!!! note
    The maximum length of `rf_dual_args` where it is highly efficient is currently 5.
    For longer length there will be a dynamic dispatch, but this number can be extended 
    by adding more methods to the *internal* `Newton.get_dual_results` function.
    
"""
function ad_newtonsolve(rf, x0, rf_dual_args; kwargs...)
    return _ad_newtonsolve(rf, x0, rf_dual_args...; kwargs...)
end

function _ad_newtonsolve(rf::F, x0::NonDualInput, rf_dual_args::DualInput...; kwargs...) where F
    rf_args = map(extract_from_dual, rf_dual_args)
    rf_singlearg(x) = rf(x, rf_args...)
    x, drdx, converged = newtonsolve(rf_singlearg, x0; kwargs...)
    drdx_inv = inv(drdx) # Not suitable for mutable functions...
    xd = get_dual_results(rf, x, rf_dual_args, rf_args, drdx_inv)
    x_value_and_dual = insert_value(xd, x) # Insert the correct value from `x` into the `xd` 
    return x_value_and_dual, converged
end

function _ad_newtonsolve(rf::F, x0::NonDualInput, rf_args::NonDualInput...; kwargs...) where F
    rf_singlearg(x) = rf(x, rf_args...)
    x, _, converged = newtonsolve(rf_singlearg, x0; kwargs...)
    return x, converged
end

extract_from_dual(v::ForwardDiff.Dual) = ForwardDiff.value(v)
extract_from_dual(t::AbstractTensor{<:Any, <:Any, <:ForwardDiff.Dual}) = Tensors._extract_value(t)
extract_from_dual(a::AbstractArray) = map(ForwardDiff.value, a)

function get_dual_results(rf::F, x, rf_dual_args::Tuple{Vararg{DualInput, 1}}, rf_args::Tuple{Vararg{NonDualInput, 1}}, drdx_inv) where F
    return seed_dual_result(rf, x, Val(1), rf_dual_args[1], rf_args, drdx_inv)
end
function get_dual_results(rf::F, x, rf_dual_args::Tuple{Vararg{DualInput, 2}}, rf_args::Tuple{Vararg{NonDualInput, 2}}, drdx_inv) where F
    xd_y₁ = seed_dual_result(rf, x, Val(1), rf_dual_args[1], rf_args, drdx_inv)
    xd_y₂ = seed_dual_result(rf, x, Val(2), rf_dual_args[2], rf_args, drdx_inv)
    return xd_y₁ + xd_y₂
end
function get_dual_results(rf::F, x, rf_dual_args::Tuple{Vararg{DualInput, 3}}, rf_args::Tuple{Vararg{NonDualInput, 3}}, drdx_inv) where F
    xd_y₁ = seed_dual_result(rf, x, Val(1), rf_dual_args[1], rf_args, drdx_inv)
    xd_y₂ = seed_dual_result(rf, x, Val(2), rf_dual_args[2], rf_args, drdx_inv)
    xd_y₃ = seed_dual_result(rf, x, Val(3), rf_dual_args[3], rf_args, drdx_inv) 
    return xd_y₁ + xd_y₂ + xd_y₃
end
function get_dual_results(rf::F, x, rf_dual_args::Tuple{Vararg{DualInput, 4}}, rf_args::Tuple{Vararg{NonDualInput, 4}}, drdx_inv) where F
    xd_y₁ = seed_dual_result(rf, x, Val(1), rf_dual_args[1], rf_args, drdx_inv)
    xd_y₂ = seed_dual_result(rf, x, Val(2), rf_dual_args[2], rf_args, drdx_inv)
    xd_y₃ = seed_dual_result(rf, x, Val(3), rf_dual_args[3], rf_args, drdx_inv)
    xd_y₄ = seed_dual_result(rf, x, Val(4), rf_dual_args[4], rf_args, drdx_inv) 
    return xd_y₁ + xd_y₂ + xd_y₃ + xd_y₄
end
function get_dual_results(rf::F, x, rf_dual_args::Tuple{Vararg{DualInput, 5}}, rf_args::Tuple{Vararg{NonDualInput, 5}}, drdx_inv) where F
    xd_y₁ = seed_dual_result(rf, x, Val(1), rf_dual_args[1], rf_args, drdx_inv)
    xd_y₂ = seed_dual_result(rf, x, Val(2), rf_dual_args[2], rf_args, drdx_inv)
    xd_y₃ = seed_dual_result(rf, x, Val(3), rf_dual_args[3], rf_args, drdx_inv)
    xd_y₄ = seed_dual_result(rf, x, Val(4), rf_dual_args[4], rf_args, drdx_inv)
    xd_y₅ = seed_dual_result(rf, x, Val(5), rf_dual_args[5], rf_args, drdx_inv) 
    return xd_y₁ + xd_y₂ + xd_y₃ + xd_y₄ + xd_y₅
end
# Fast up to 5 args, then slower (dispatch...)

function get_dual_results(rf::F, x, rf_dual_args::Tuple{Vararg{DualInput, N}}, rf_args::Tuple{Vararg{NonDualInput, N}}, drdx_inv) where {F, N}
    return sum(i -> seed_dual_result(rf, x, Val(i), rf_dual_args[i], rf_args, drdx_inv), 1:N)
end

function seed_dual_result(rf::F, x, ::Val{I}, dual_arg, rf_args, drdx_inv) where {F, I}
    r_dual_yᵢ = rf(x, rf_args[1:(I-1)]..., dual_arg, rf_args[(I+1):end]...)
    x_dual_yᵢ = - appropriate_contraction(drdx_inv, r_dual_yᵢ) # Linear operation, values in x not important. 
    return return x_dual_yᵢ
end

appropriate_contraction(A::AbstractTensor{2}, b::AbstractTensor{1}) = A ⋅ b
appropriate_contraction(A::AbstractTensor{4}, b::AbstractTensor{2}) = A ⊡ b 
appropriate_contraction(A::SMatrix, b::SVector) = A * b 
appropriate_contraction(A::Number, b::Number) = A * b 

insert_value(x_dual_yᵢ::TD, x::T) where {T, TD <: ForwardDiff.Dual{<:Any, T}} = TD(x, ForwardDiff.partials(x_dual_yᵢ))
insert_value(x_dual_yᵢ::StaticArray, x::StaticArray) = map(insert_value, x_dual_yᵢ, x)
function insert_value(x_dual_yᵢ::TTD, x::TT) where 
    {TTD <: Tensors.AbstractTensor{order, dim, TD}, 
    TT   <: Tensors.AbstractTensor{order, dim, <:Number}} where {TD <: ForwardDiff.Dual, order, dim}
    return TTD(map(insert_value, Tensors.get_data(x_dual_yᵢ), Tensors.get_data(x)))
end
