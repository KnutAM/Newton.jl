using Newton
using Test
using ForwardDiff, FiniteDiff
using LinearAlgebra
using StaticArrays
using Tensors
using Test

include("test_newtonsolver.jl")
include("test_inv.jl")
include("test_ad_solver.jl")
include("test_logging.jl")

@testset "linsolve!" begin
    A = 2*I + rand(10,10)
    b = rand(10)
    A1, A2 = [copy(A) for _ in 1:2]
    b1, b2 = [copy(b) for _ in 1:2]
    rf!(r, x) = (r .= x)
    @test A1\b1 ≈ Newton.linsolve!(A2, b2, NewtonCache(b))
end