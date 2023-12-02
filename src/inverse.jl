"""
    Newton.inv!(A::Matrix, cache::NewtonCache)

Utilize the LU decomposition from `RecursiveFactorization.jl` along with 
the non-exported `LinearAlgebra.inv!(::LU)` to calculate the inverse of 
`A` more efficient than `inv(A)`. Note that `A` will be used as workspace
and values should not be used after calling `Newton.inv!`
"""
function inv!(A::Matrix, cache::NewtonCache)
    luA = RecursiveFactorization.lu!(A, cache.lupivot, #=pivot=#Val(true), #=thread=#Val(false))
    return LinearAlgebra.inv!(luA)
end