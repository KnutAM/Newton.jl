"""
Newton.jl comes with the following linear solvers

* [StandardLinsolver](@ref) (default)
* [UnsafeFastLinsolver](@ref)
* [RecursiveFactorizationLinsolver](@ref)

"""
abstract type AbstractLinsolver end

"""
    Newton.StandardLinsolver()

This is the default linear solver, which gives safe operations and don't require any special packages to be loaded.
"""
struct StandardLinsolver end
default_linsolver(::Any) = StandardLinsolver()

"""
    UnsafeFastLinsolver()

This is a special linear solver, which calculates the inverse recursively by using the analytical inverses of 2x2 and 3x3
matrices. This gives exceptional performance for small matrices, but suffers from numerical errors and can be sensitive to 
badly conditioned matrices. When using this method, it may be advisable to (adaptively) try a slower method if the newton 
iterations fail to converge.
"""
struct UnsafeFastLinsolver end

# Behaviour of which defined in extensions:

"""
    RecursiveFactorizationLinsolver()

This linear solver utilizes the LU decomposition in `RecursiveFactorization.jl`, which gives faster performance than the 
`StandardLinsolver`. While not as fast as `UnsafeFastLinsolver`, it is always accurate. Is available via an extension, 
requiring the user to load `RecursiveFactorization.jl` separately. 
"""
struct RecursiveFactorizationLinsolver
    function RecursiveFactorizationLinsolver()
        if !hasmethod(linsolve!, Tuple{RecursiveFactorizationLinsolver, Matrix, Vector, NewtonCache})
            error("Please call `using RecursiveFactorization` if we you want to use this solver")
        end
        return new()
    end
end


"""
    linsolve!(K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)

Solves the linear equation system `Kx=b`, mutating both `K` and `b`.
`b` is mutated to the solution `x`.

    linsolve!(linsolver, K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)

The default implementation will call this signature, which should be overloaded for a different 
`linsolver` passed to `cache` upon construction.
"""
function linsolve!(K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)
    return linsolve!(cache.linsolver, K, b, cache)
end

# StandardLinsolver
function linsolve!(::StandardLinsolver, K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)
    ldiv!(lu!(K), b)
    return b
end

# UnsafeFastLinsolver
function linsolve!(::UnsafeFastLinsolver, K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)
    if length(b) < 50 # Accuracy can become really unreliable for larger matrices, and no performance improvement either. 
        sinv!(K)
        return K * b
    else
        return linsolve!(StandardLinsolver(), K, b, cache)
    end
end

"""
    linsolve(linsolver, K, b)

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

# StandardLinsolver
@inline linsolve(::StandardLinsolver, K::Number, b::Number) = b / K
@inline linsolve(::StandardLinsolver, K::SMatrix, b::SVector) = K \ b
@inline linsolve(::StandardLinsolver, K::SecondOrderTensor, b::AbstractTensor) = K \ b 
@inline linsolve(::StandardLinsolver, K::FourthOrderTensor, b::AbstractTensor) = inv(K) ⊡ b

# UnsafeFastLinsolver
@inline linsolve(::UnsafeFastLinsolver, K::Number, b::Number) = b / K
@inline linsolve(::UnsafeFastLinsolver, K::SMatrix, b::SVector) = sinv(K) * b
@inline linsolve(::UnsafeFastLinsolver, K::SecondOrderTensor, b::AbstractTensor) = K \ b # Tensors.jl has a fast implementation, using that. 
@inline function linsolve(::UnsafeFastLinsolver, K::FourthOrderTensor, b::AbstractTensor)
    return frommandel(Tensors.get_base(typeof(K)), sinv(tomandel(SArray, K))) ⊡ b
end
