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

# The following optimization only works on 1.9
import LinearAlgebra as LA

@static if VERSION > v"1.9"

    function _inv!(A::LU{T,<:StridedMatrix}, cache::NewtonCache{T}) where {T<:Float64}
        Adata = getproperty(A, :factors)
        ipiv = getproperty(A, :ipiv)
        return lapack_getri!(Adata, ipiv, cache.blas_work)
    end

    # Code from LinearAlgebra/src/lapack.jl, adjusted to allow work::Vector as input 
    function lapack_getri!(A::AbstractMatrix{T}, ipiv::AbstractVector{LA.BlasInt}, work::Vector{T}) where {T<:Float64} 
        LA.require_one_based_indexing(A, ipiv)
        LA.chkstride1(A, ipiv)
        n = LA.checksquare(A)
        if n != length(ipiv)
            throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $n"))
        end
        lda = max(1,stride(A, 2))
        info  = Ref{LA.BlasInt}()

        if length(work) < max(1, n)
            lwork = LA.BlasInt(-1)
            length(work) == 0 && resize!(work, 1)
            ccall((LA.BLAS.@blasfunc(dgetri_), LA.libblastrampoline), Cvoid,
                (Ref{LA.BlasInt}, Ptr{T}, Ref{LA.BlasInt}, Ptr{LA.BlasInt},
                Ptr{T}, Ref{LA.BlasInt}, Ptr{LA.BlasInt}),
                n, A, lda, ipiv, work, lwork, info)
            LA.LAPACK.chklapackerror(info[])
            lwork = LA.BlasInt(real(work[1]))
            resize!(work, lwork)
        else
            lwork = LA.BlasInt(length(work))
        end
        
        ccall((LA.BLAS.@blasfunc(dgetri_), LA.libblastrampoline), Cvoid,
                (Ref{LA.BlasInt}, Ptr{T}, Ref{LA.BlasInt}, Ptr{LA.BlasInt},
                Ptr{T}, Ref{LA.BlasInt}, Ptr{LA.BlasInt}),
                n, A, lda, ipiv, work, lwork, info)
        LA.LAPACK.chklapackerror(info[])
        lwork = LA.BlasInt(real(work[1]))
        if length(work) != lwork
            resize!(work, lwork)
        end
        return A
    end

end