using BenchmarkTools
using LinearAlgebra
using Newton
using NLsolve
using StaticArrays

LinearAlgebra.BLAS.set_num_threads(1) # Otherwise very slow...

rf(x::Number) = exp(x) - x^2

function rf(x::SVector)
    return exp.(x) - x.^2
end

function rf!(r::Vector, x::Vector)
    return map!(v->(exp(v)-v^2), r, x)
end

function run_benchmark(dim)
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
    @btime newtonsolve(x_, $rf) setup=(x_=copy($x_s)) evals=1;

    # Get time for newtonsolve!
    # First time setup
    cache = NewtonCache(x_d, rf!)
    
    print("newtonsolve dynamic: ")
    @btime newtonsolve(x_, $rf!, $cache) setup=(x_=copy($x_d)) evals=1;

    # Get time for nlsolve
    print("nlsolve dynamic:     ")
    @btime nlsolve($rf!, x_, autodiff=:forward) setup=(x_=copy($x_d)) evals=1;

    println("")
end

x = 0.0
print("rf (scalar):         ")
@btime rf($x)
print("newtonsolve scalar:  ")
@btime newtonsolve($x, $rf)    

run_benchmark.([5,10,20,40])