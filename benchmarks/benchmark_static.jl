using BenchmarkTools
using LinearAlgebra
using Newton
using NLsolve
using StaticArrays

LinearAlgebra.BLAS.set_num_threads(1)

function rf(x::SVector)
    return exp.(x) - x.^2
end

function rf!(r::Vector, x::Vector)
    @simd for i in 1:length(x)
        @inbounds r[i] = exp(x[i]) - x[i]^2
    end
    return r
end

function mn!(the_x, the_drdx, the_rf!, the_cache)
    println(size.((the_x, the_drdx)))
    return newtonsolve!(the_x, the_drdx, the_rf!, the_cache)
end

function run_benchmark(dims)

    for dim = dims

        println("Benchmark with dim=$dim")
        x_s = zero(SVector{dim})
        x_d = zeros(dim)
        y_d = zeros(dim)

        # Check that rf/rf! is non-allocating
        print("rf (static):         ")
        @btime rf($x_s)
        print("rf (dynamic):        ")
        @btime rf!($y_d, $x_d)

        # Get time for newtonsolve
        print("newtonsolve static:  ")
        @btime newtonsolve($x_s, $rf);

        # Get time for newtonsolve!
        # First time setup
        cache = NewtonCache(x_d, rf!)
        drdx = get_drdx(cache)
        print("newtonsolve dynamic: ")
        @btime newtonsolve!(x_, $drdx, $rf!, $cache) setup=(x_=copy($x_d)) evals=1;

        # Get time for nlsolve
        print("nlsolve dynamic:     ")
        @btime nlsolve($rf!, x_, autodiff=:forward) setup=(x_=copy($x_d)) evals=1;

        println("")
    end
end

run_benchmark([5,10,20])