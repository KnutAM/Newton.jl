"""
    Newton.inv!(A::Matrix, cache::NewtonCache)

In-place inverse, which, depending on the `linsolver` in `cache`, can be much more 
efficient than `inv(A)`. However, note that `A` will be used as workspace
and values should not be used after calling `Newton.inv!`. In some cases, 
`A` will become its inverse, and the output aliased to `A`. 
This behavior is not true in general, and should not be relied upon. 
"""
@inline inv!(A::AbstractMatrix, c::NewtonCache) = inv!(c.linsolver, A, c)

@inline function inv!(::StandardLinsolver, A::Matrix, c::NewtonCache)
    f = LinearAlgebra.lu!(A)
    return _inv!(f, c)
end

@inline function inv!(::UnsafeFastLinsolver, A::AbstractMatrix, ::NewtonCache)
    return sinv!(A)
end

# Implementation of in-place `inv!(::LU, cache)` function

# Generic fallback 
_inv!(A::LU, ::NewtonCache) = LinearAlgebra.inv!(A)

# Optimized that only works on Julia 1.9 or later
import LinearAlgebra as LA

@static if VERSION ≥ v"1.9"

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

# StaticArrays fast inverse implementation: sinv(::SMatrix{d, d}) where d
# Seems to work well up to 50x50, after which the error for random matrices seems 
# to increase. Speedup until about 60x60.

"""
    sinv(a::SMatrix{d, d}) where {d}

Fast, but numerically unstable implementation of the inverse of a statically sized matrix.
Beneficial up to d ≈ 50, but can give large floating point errors for badly conditioned and/or large matrices.

About 4 to 5 timers faster than StaticArrays for sizes ∈ [5, 20], and at least twice as fast up to 50 according 
to benchmarks on macbook with M3 processor.
"""
function sinv end

@inline sinv(a::SMatrix{1, 1}) = inv(a)
@inline sinv(a::SMatrix{2, 2}) = inv(a)
@inline sinv(a::SMatrix{3, 3}) = inv(a)

const _SM{d} = SMatrix{d, d}

@inline function sinv(a::Union{_SM{4}, _SM{5}, _SM{6}, _SM{7}, _SM{8}, _SM{9}, _SM{10}})
    getd(::SMatrix{dd, dd}) where {dd} = dd
    d = getd(a)
    d1 = d ÷ 2
    d2 = d - d1
    b11, b21, b12, b22 = solve_pairwise(a, Val(d1), Val(d2))
    return join_submatrices(b11, b21, b12, b22)
end

function sinv(a::SMatrix{d, d}) where {d}
    d1 = d ÷ 2
    d2 = d - d1
    b11, b21, b12, b22 = solve_pairwise(a, Val(d1), Val(d2))
    return join_submatrices(b11, b21, b12, b22)
end

"""
    extract_submatrix(::Type{SMatrix{d1, d2}}, m::SMatrix, start_row, start_col)

Efficiently extract `s::SMatrix{d1, d2}` such that 
`s == m[start_row:(start_row + d1 - 1), start_col:(start_col + d2 - 1)]`
"""
@generated function extract_submatrix(::Type{SMatrix{d1, d2}}, m::SMatrix, start_row::Int, start_col::Int) where {d1, d2}
    ex = :(SMatrix{$d1, $d2}())
    for col_offset in 0:(d2 - 1)
        for row_offset in 0:(d1 - 1)
            push!(ex.args, :(m[start_row + $row_offset, start_col + $col_offset]))
        end
    end
    quote
        @inbounds return $ex
    end
end

"""
    join_submatrices(a11, a12, a21, a22)

Efficiently join the submatrices to return a::SMatrix = [a11 a12; a21 a22].
"""
@generated function join_submatrices(a11::SMatrix{r1, c1}, a21::SMatrix{r2, c1}, a12::SMatrix{r1, c2}, a22::SMatrix{r2, c2}) where {r1, r2, c1, c2}
    r = r1 + r2
    c = c1 + c2
    ex = :(SMatrix{$r, $c}())
    for col in 1:c
        for row in 1:r
            t = if row ≤ r1 && col ≤ c1
                :(a11[$row, $col])
            elseif row > r1 && col ≤ c1
                :(a21[$(row - r1), $col])
            elseif row ≤ r1 && col > c1
                :(a12[$row, $(col - c1)])
            else # row > r1 && col > c1
                :(a22[$(row - r1), $(col - c1)])
            end
            push!(ex.args, t)
        end
    end
    quote
        @inbounds return $ex
    end
end


@inline function solve_pairwise(a::SMatrix{d, d}, ::Val{d1}, ::Val{d2}) where {d, d1, d2}
    @assert d == d1 + d2
    @inbounds begin                                                 # E.g. for d1=d2=2
        a11 = extract_submatrix(SMatrix{d1, d1}, a, 1, 1)           # a[1:2, 1:2]
        a21 = extract_submatrix(SMatrix{d2, d1}, a, d1 + 1, 1)      # a[3:4, 1:2]
        a12 = extract_submatrix(SMatrix{d1, d2}, a, 1, d1 + 1)      # a[1:2, 3:4]
        a22 = extract_submatrix(SMatrix{d2, d2}, a, d1 + 1, d1 + 1) # a[3:4, 3:4]

        a22_inv_times_a21 = sinv(a22) * a21
        a11_inv_times_a12 = sinv(a11) * a12

        b11 = sinv(a11 - a12 * a22_inv_times_a21)
        b22 = sinv(a22 - a21 * a11_inv_times_a12)
        b12 = -a11_inv_times_a12 * b22
        b21 = -a22_inv_times_a21 * b11
        
        return b11, b21, b12, b22
    end
end

"""
    sinv!(K::Matrix)

Invert `K` in-place using the unsafe static implementation up to a size of 20x20,
and fall back to generic `LinearAlgebra.inv`
"""
@inline function sinv!(K::Matrix)
    n = size(K, 1)
    @assert n == size(K, 2) > 0

    @inbounds begin
        if n == 1
            K .= 1 / K[1,1]
        elseif n == 2
            K .= sinv(SMatrix{2, 2}(K))
        elseif n == 3
            K .= sinv(SMatrix{3, 3}(K))
        elseif n == 4
            K .= sinv(SMatrix{4, 4}(K))
        elseif n == 5
            K .= sinv(SMatrix{5, 5}(K))
        elseif n == 6
            K .= sinv(SMatrix{6, 6}(K))
        elseif n == 7
            K .= sinv(SMatrix{7, 7}(K))
        elseif n == 8
            K .= sinv(SMatrix{8, 8}(K))
        elseif n == 9
            K .= sinv(SMatrix{9, 9}(K))
        elseif n == 10
            K .= sinv(SMatrix{10, 10}(K))
        else # n ≥ 11
            sinv_11!(K, n)
        end
    end
    return K
end

function sinv_11!(K, n)
    @inbounds begin
        if n == 11
            K .= sinv(SMatrix{11, 11}(K))
        elseif n == 12
            K .= sinv(SMatrix{12, 12}(K))
        elseif n == 13
            K .= sinv(SMatrix{13, 13}(K))
        elseif n == 14
            K .= sinv(SMatrix{14, 14}(K))
        elseif n == 15
            K .= sinv(SMatrix{15, 15}(K))
        elseif n == 16
            K .= sinv(SMatrix{16, 16}(K))
        elseif n == 17
            K .= sinv(SMatrix{17, 17}(K))
        elseif n == 18
            K .= sinv(SMatrix{18, 18}(K))
        elseif n == 19
            K .= sinv(SMatrix{19, 19}(K))
        elseif n == 20
            K .= sinv(SMatrix{20, 20}(K))
        else
            # Shouldn't be used for this case, but to give the right result...
            K .= LinearAlgebra.inv!(LinearAlgebra.lu!(K))
        end
    end
    return K
end


