using BenchmarkTools
using LinearAlgebra
using Newton
using NLsolve
using StableRNGs

LinearAlgebra.BLAS.set_num_threads(1)

function create_rf!(dim, rng=StableRNG(0))
    A = -rand(rng, dim, dim)
    b = -rand(rng, dim)

    function rf!(r, x)
        r .= b
        mul!(r, A, x, 1, 1)
        @simd for i=1:length(x)
            @inbounds r[i] += exp(x[i]) - x[i]^2
        end
        return r
    end

    return rf!
end

dim = 20
rf = create_rf!(dim)
r, x = [zeros(dim) for _ in 1:2]

# Check that rf! is non-allocating
println("@btime rf!")
@btime rf(r_, x_) setup=(r_=copy(r); x_=copy(x)) evals=1;

# Get time for newtonsolve!
# First time setup
cache = NewtonCache(x, rf)
drdx = get_drdx(cache)
println("@btime newtonsolve!")
@btime newtonsolve!(x_, $drdx, rf, $cache) setup=(x_=copy(x)) evals=1;

# Get time for nlsolve
println("@btime nlsolve")
@btime nlsolve(rf, x_, autodiff=:forward) setup=(x_=copy(x)) evals=1;

println("Benchmark (dim=$dim) complete")