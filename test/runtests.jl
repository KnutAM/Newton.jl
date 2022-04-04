using Newton
using Test
using ForwardDiff
using LinearAlgebra
using StaticArrays

multiinput_rf!(r::Vector, x::Vector, A::Matrix, b::Vector) = (r .= b .+ A*x)

function setup_cache(x, A, b)
    rf!(r, x) = multiinput_rf!(r, x, A, b)
    return NewtonCache(x, rf!)
end

function determine_solution(A, b)
    x = zero(b)
    rf!(r, x) = multiinput_rf!(r, x, A, b)
    cache=NewtonCache(x, rf!)
    newtonsolve!(x, get_drdx(cache), rf!, cache)
    return x
end

@testset "linsolve!" begin
    A = 2*I + rand(10,10)
    b = rand(10)
    A1, A2 = [copy(A) for _ in 1:2]
    b1, b2 = [copy(b) for _ in 1:2]
    rf!(r, x) = (r .= x)
    @test A1\b1 ≈ Newton.linsolve!(A2, b2, NewtonCache(b, rf!))
end

@testset "newtonsolve!" begin
    nsize = 4
    (a,b,x0) = [rand(nsize) for _ in 1:3]
    tol = 1.e-10
    
    # Basic functionality
    function rf_solution!(r, x)
        r .= - a + b.*x + exp.(x)
    end
    x = copy(x0)
    cache = NewtonCache(x, rf_solution!)
    drdx = get_drdx(cache)
    @test size(drdx,1) == nsize
    converged = newtonsolve!(x, drdx, rf_solution!; tol=tol)
    r_check = similar(x0)
    @test converged
    @test isapprox(rf_solution!(r_check, x), zero(r_check); atol=tol)
    @test drdx ≈ ForwardDiff.jacobian(rf_solution!, r_check, x)

    function rf_nosolution!(r, x)
        r .= a .+ b.*x.^2
    end
    x = copy(x0)
    converged = newtonsolve!(x, drdx, rf_nosolution!)
    @test !converged
    
    # Test function with different anynomous functions for cache and solving
    cache = setup_cache(x0, zeros(nsize,nsize), zeros(nsize))   # If these A and b inputs are used, solution is zero
    A = nsize*I+rand(nsize,nsize)
    A += transpose(A)   # Symmetrize to ensure invertible
    b = rand(nsize)
    
    rf_solve!(r, x) = multiinput_rf!(r, x, A, b)
    drdx = get_drdx(cache)
    @test newtonsolve!(x, drdx, rf_solve!, cache)
    @test isapprox(rf_solve!(r_check, x), zero(r_check); atol=tol)
    @test .!(isapprox(x, zero(x); atol=tol))

    # Test that error is thrown if we try to use automatic differentiation of the newtonsolve!
    diff_fun(y) = determine_solution(A, y)
    failed = false
    try
        df = ForwardDiff.jacobian(diff_fun, rand(nsize))
    catch err
        failed = true
    end
    @test failed

end

@testset "linsolve" begin
    dim = 5
    A = rand(SMatrix{dim,dim})
    x = rand(SVector{dim})
    b = A*x
    @test x ≈ Newton.linsolve(A, b)
end

@testset "newtonsolve" begin
    dim = 6
    (a,b,x0) = [rand(SVector{dim}) for _ in 1:3]

    rf_solution(x) = - a + b.*x + exp.(x)
    rf_nosolution(x) = a .+ b.*x.^2

    converged, x, drdx = newtonsolve(x0, rf_solution; tol=1.e-6)
    @test converged
    @test isapprox(norm(rf_solution(x)), 0.0, atol=1.e-6)
    @test drdx ≈ ForwardDiff.jacobian(rf_solution, x)

    converged, x, drdx = newtonsolve(x0, rf_nosolution; tol=1.e-6, maxiter=4)
    @test ~converged
    @test all(isnan.(x))
    @test all(isnan.(drdx))

end