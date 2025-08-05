using Newton
using RecursiveFactorization
using LinearAlgebra
using BenchmarkTools
using StaticArrays
import CairoMakie as Plt
using AppleAccelerate

# Test functions 
# 1) testlu!(cache, ::AbstractMatrix) # if possible
# 2) testinv!(cache, ::AbstractMatrix)
# 3) testsolve!(cache, ::AbstractMatrix, ::AbstractVector)

struct RFCache{T}
    lupivot::Vector{Int}
    blaswork::Vector{T}
end
RFCache(x::Vector) = RFCache(Vector{Int}(undef, length(x)), similar(x, 0))

function testlu!(cache::RFCache, K::Matrix)
    return RecursiveFactorization.lu!(K, cache.lupivot, Val{true}(), Val{false}())
end

function testinv!(cache::RFCache, K::Matrix)
    return _inv!(testlu!(cache, K), cache)
end

function _inv!(A::LU{T,<:StridedMatrix}, cache::RFCache{T}) where {T<:Float64}
    Adata = getproperty(A, :factors)
    ipiv = getproperty(A, :ipiv)
    return Newton.lapack_getri!(Adata, ipiv, cache.blaswork)
end

function testsolve!(cache::RFCache, K::Matrix, f::Vector)
    return testlu!(cache, K) \ f
end

struct StaticCache{N} end

function testinv!(::StaticCache{N}, K::Matrix) where {N}
    S = SMatrix{N, N}(K)
    return inv(S)
end

function testsolve!(::StaticCache{N}, K::Matrix, f::Vector) where {N}
    S = SMatrix{N, N}(K)
    g = SVector{N}(f)
    return S \ g
end

struct Standard end

testlu!(::Standard, K::Matrix) = lu!(K)

testinv!(::Standard, K::Matrix) = inv(K)

testsolve!(::Standard, K::Matrix, f::Vector) = K \ f

struct Standard2 end

testlu!(::Standard2, K::Matrix) = lu!(K)

testinv!(::Standard2, K::Matrix) = inv(testlu!(Standard(), K))

testsolve!(::Standard2, K::Matrix, f::Vector) = testlu!(Standard(), K) \ f


setup_fun(n) = (rand(n, n) + 10I, rand(n))



cases = (   "Standard" => n -> Standard(), 
            "Standard2" => n -> Standard2(), 
            "StaticCache" => n -> StaticCache{n}(), 
            "RFCache" => n -> RFCache(rand(n))
            )

function timetest(cases, r = 1:6)
    sizes = Int[]
    invtime = Dict{String, Vector{Float64}}()
    soltime = Dict{String, Vector{Float64}}()
    for nroot in r
        n = nroot^2
        push!(sizes, n)
        @show n
        for (name, getcache) in cases
            cache = getcache(n)
            tinv = get!(invtime, name, Float64[])
            push!(tinv, minimum((@benchmark testinv!($cache, K) setup = ((K, f) = setup_fun($n))).times))
            tsol = get!(soltime, name, Float64[])
            push!(tsol, minimum((@benchmark testsolve!($cache, K, f) setup = ((K, f) = setup_fun($n))).times))
        end
    end
    return sizes, invtime, soltime
end

function plottest(cases, sizes, invtime, soltime)
    fig = Plt.Figure()
    Plt.Label(fig[1,1], "inverse", tellwidth=false)
    Plt.Label(fig[1,2], "solution", tellwidth=false)
    ax1 = Plt.Axis(fig[2,1]; xlabel = "size", ylabel = "time [ns]", yscale = log10)
    ax2 = Plt.Axis(fig[2,2]; xlabel = "size", ylabel = "time [ns]", yscale = log10)
    for (key, _) in cases
        Plt.lines!(ax1, sizes, invtime[key]; label = key)
        Plt.lines!(ax2, sizes, soltime[key]; label = key)
    end
    Plt.axislegend(ax1; position = :rb)
    Plt.axislegend(ax2; position = :rb)

    return fig
end