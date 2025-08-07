module RecursiveFactorizationExt

using RecursiveFactorization: RecursiveFactorization
using Newton: Newton, RecursiveFactorizationLinsolver, NewtonCache
using LinearAlgebra: LinearAlgebra

@inline function Newton.linsolve!(::RecursiveFactorizationLinsolver, K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)
    LU = RecursiveFactorization.lu!(K, cache.lupivot, Val{true}(), Val{false}())
    LinearAlgebra.ldiv!(LU, b)
    return b
end

@inline function Newton.inv!(::RecursiveFactorizationLinsolver, A::Matrix, cache::NewtonCache)
    luA = RecursiveFactorization.lu!(A, cache.lupivot, #=pivot=#Val(true), #=thread=#Val(false))
    return Newton._inv!(luA, cache)
end

end