using Newton
using Test
using ForwardDiff, FiniteDiff
using LinearAlgebra
using StaticArrays
using Tensors
using Test

# Test error msg with extensions before loading extension packages
@test_throws ErrorException Newton.RecursiveFactorizationLinsolver()
@test_throws "using RecursiveFactorization" Newton.RecursiveFactorizationLinsolver()

using RecursiveFactorization

if !Newton.LOGGING    
    include("test_inv.jl")
    include("test_linsolvers.jl")
    include("test_newtonsolver.jl")
    include("test_ad_in_residual.jl")
    include("test_ad_solver.jl")
    include("test_deprecated.jl")
else
    include("test_logging.jl")      # Specific checks
    include("test_newtonsolver.jl") # Integration tests are done also in logging mode. 
end
