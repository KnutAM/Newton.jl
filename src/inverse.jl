"""
    Newton.inv!(A::Matrix, cache::NewtonCache)

Utilize the LU decomposition from `RecursiveFactorization.jl` along with 
the non-exported `LinearAlgebra.inv!(::LU)` to calculate the inverse of 
`A` more efficient than `inv(A)`. Note that `A` will be used as workspace
and values should not be used after calling `Newton.inv!`
"""
function inv!(A::Matrix, cache::NewtonCache)
    luA = RecursiveFactorization.lu!(A, cache.lupivot, #=pivot=#Val(true), #=thread=#Val(false))
    return _inv!(luA, cache)
end

_inv!(A::LU, ::NewtonCache) = LinearAlgebra.inv!(A)

function _inv!(A::LU{T,<:StridedMatrix}, cache::NewtonCache{T}) where {T<:Float64}
    Adata = getproperty(A, :factors)
    ipiv = getproperty(A, :ipiv)
    return lapack_getri!(Adata, ipiv, cache.blas_work)
end

import LinearAlgebra: libblastrampoline, require_one_based_indexing, BlasInt, chkstride1, checksquare
import LinearAlgebra.BLAS: @blasfunc
import LinearAlgebra.LAPACK: chklapackerror

# Code from LinearAlgebra/src/lapack.jl, adjusted to allow work::Vector as input 
function lapack_getri!(A::AbstractMatrix{T}, ipiv::AbstractVector{BlasInt}, work::Vector{T}) where {T<:Float64} 
    require_one_based_indexing(A, ipiv)
    chkstride1(A, ipiv)
    n = checksquare(A)
    if n != length(ipiv)
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $n"))
    end
    lda = max(1,stride(A, 2))
    info  = Ref{BlasInt}()

    if length(work) < max(1, n)
        lwork = BlasInt(-1)
        length(work) == 0 && resize!(work, 1)
        ccall((@blasfunc(dgetri_), libblastrampoline), Cvoid,
              (Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt},
               Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}),
              n, A, lda, ipiv, work, lwork, info)
        chklapackerror(info[])
        lwork = BlasInt(real(work[1]))
        resize!(work, lwork)
    else
        lwork = BlasInt(length(work))
    end
    
    
    ccall((@blasfunc(dgetri_), libblastrampoline), Cvoid,
            (Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{BlasInt},
            Ptr{T}, Ref{BlasInt}, Ptr{BlasInt}),
            n, A, lda, ipiv, work, lwork, info)
    chklapackerror(info[])
    lwork = BlasInt(real(work[1]))
    if length(work) != lwork
        resize!(work, lwork)
    end
    return A
end