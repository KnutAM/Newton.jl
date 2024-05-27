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
    linsolve(K, b)

Solve the equation (system) `b = K ⋆ x` where `⋆` is the appropriate contraction between `K` and `x`,
and `b` and `x` have the same type and size. The following combinations of `K` and `x` are supported. 

| K                      | `b`, `x`               |
| ---------------------- | ---------------------- |
| `Number`               | `Number`               |
| `SMatrix{N, N}`        | `SVector{N}`           |
| `SecondOrderTensor{d}` | `AbstractTensor{o, d}` |
| `FourthOrderTensor{d}` | `AbstractTensor{o, d}` | 

"""
function linsolve end


linsolve(K::Number, b::Number) = b / K
linsolve(K::SMatrix, b::SVector) = K \ b
linsolve(K::SecondOrderTensor, b::AbstractTensor) = K \ b 
linsolve(K::FourthOrderTensor, b::AbstractTensor) = inv(K) ⊡ b
