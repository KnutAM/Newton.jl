using StaticArrays
using BenchmarkTools
import CairoMakie as Plt

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

@inline sinv(a::SMatrix{1, 1}; kwargs...) = inv(a)
@inline sinv(a::SMatrix{2, 2}; kwargs...) = inv(a)
@inline sinv(a::SMatrix{3, 3}; kwargs...) = inv(a)

function sinv(a::SMatrix{d, d}) where {d}
    d1 = d ÷ 2
    d2 = d - d1
    b11, b21, b12, b22 = solve_pairwise(a, Val(d1), Val(d2))
    return join_submatrices(b11, b21, b12, b22)
end

#=
function sinv(a::SMatrix{4, 4})
    b11, b21, b12, b22 = solve_pairwise(a, Val(2), Val(2))
    return join_submatrices(b11, b21, b12, b22)
end

function sinv(a::SMatrix{5, 5})
    b11, b21, b12, b22 = solve_pairwise(a, Val(2), Val(3))
    return join_submatrices(b11, b21, b12, b22)
end

function sinv(a::SMatrix{6, 6})
    b11, b21, b12, b22 = solve_pairwise(a, Val(3), Val(3))
    return join_submatrices(b11, b21, b12, b22)
end

function sinv(a::SMatrix{7, 7})
    b11, b21, b12, b22 = solve_pairwise(a, Val(3), Val(4))
    return join_submatrices(b11, b21, b12, b22)
end
=#



function runit(v::Vector{<:SMatrix})
    s = zero(eltype(v))
    for x in v
        si = sinv(x)
        s = s + si
    end
    return s
end

function timeit()
    dims = [1:20..., 25:5:60...]
    inv_time = Float64[]
    sinv_time = Float64[]
    rinv_time = Float64[]
    for n in dims
        a = rand(SMatrix{n, n})
        println("SMatrix{$n, $n}: Equality check passed: ", inv(a) ≈ sinv(a))
        push!(inv_time,  minimum((@benchmark  inv($(rand(SMatrix{n, n})))).times))
        push!(sinv_time, minimum((@benchmark sinv($(rand(SMatrix{n, n})))).times))
        cache = RFCache(zeros(n))
        push!(rinv_time, minimum((@benchmark testinv!($cache, $(rand(n, n)))).times))
        println("inv: ", last(inv_time), ", sinv: ", last(sinv_time), ", speedup: ", (last(inv_time), last(rinv_time)) ./ last(sinv_time))
        println()
    end
    return dims, inv_time, sinv_time, rinv_time
end

function plotit(dims, inv_time, sinv_time, rinv_time)
    fig = Plt.Figure()
    ax = Plt.Axis(fig[1,1]; xlabel = "dim", ylabel = "time [ns]", yscale = log10)
    Plt.lines!(ax, dims, inv_time; label = "inv")
    Plt.lines!(ax, dims, rinv_time; label = "rinv")
    Plt.lines!(ax, dims, sinv_time; label = "sinv")
    Plt.axislegend(ax; position = :rb)
    ax2 = Plt.Axis(fig[1,2]; xlabel = "dim", ylabel = "speedup")
    Plt.lines!(ax2, dims, inv_time ./ sinv_time; label = "inv")
    Plt.lines!(ax2, dims, rinv_time ./ sinv_time; label = "rinv")
    return fig
end
