using Newton
using Test
using ForwardDiff
using LinearAlgebra
using StaticArrays

include("test_logging.jl")

multiinput_rf!(r::Vector, x::Vector, A::Matrix, b::Vector) = (r .= b .+ A*x)

function setup_cache(x, A, b)
    rf!(r, x) = multiinput_rf!(r, x, A, b)
    return NewtonCache(x, rf!)
end

function determine_solution(A, b)
    x0 = zero(b)
    rf!(r, x) = multiinput_rf!(r, x, A, b)
    cache=NewtonCache(x0, rf!)
    x, _, _ = newtonsolve(x0, rf!, cache)
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

@testset "newtonsolve (dynamic)" begin
    nsize = 4
    (a,b,x0) = [rand(nsize) for _ in 1:3]
    tol = 1.e-10
    
    # Basic functionality
    function rf_solution!(r, x)
        r .= - a + b.*x + exp.(x)
    end
    cache = NewtonCache(x0, rf_solution!)
    @test x0 !== getx(cache) # Check that x0 is not aliased to getx(cache)
    xguess = getx(cache)
    copy!(xguess, x0)
    x, drdx, converged = newtonsolve(xguess, rf_solution!; tol=tol)
    r_check = similar(x0)
    @test x0 == xguess     # Input should not be modified when not aliased
    @test converged
    @test isapprox(rf_solution!(r_check, x), zero(r_check); atol=tol)
    @test drdx ≈ ForwardDiff.jacobian(rf_solution!, r_check, x)

    # Test with given cache
    copy!(getx(cache), x0)
    x, drdx, converged = newtonsolve(getx(cache), rf_solution!, cache; tol=tol)
    @test x === getx(cache) # Output should be aliased to cache

    function rf_nosolution!(r, x)
        r .= a .+ b.*x.^2
    end
    x = copy(x0)
    x, drdx, converged = newtonsolve(x, rf_nosolution!)
    @test !converged
    
    # Test function with different anynomous functions for cache and solving
    cache = setup_cache(x0, zeros(nsize,nsize), zeros(nsize))   # If these A and b inputs are used, solution is zero
    A = nsize*I+rand(nsize,nsize)
    A += transpose(A)   # Symmetrize to ensure invertible
    b = rand(nsize)
    
    x = copy(x0)
    rf_solve!(r, x) = multiinput_rf!(r, x, A, b)
    x, drdx, converged = newtonsolve(x, rf_solve!, cache)
    @test converged
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

@testset "newtonsolve (static)" begin
    dim = 6
    for T in (Float32,Float64)    
        (a,b,x0) = [rand(SVector{dim,T}) for _ in 1:3]

        rf_solution(x) = - a + b.*x + exp.(x)
        rf_nosolution(x) = a .+ b.*x.^2

        x, drdx, converged = newtonsolve(x0, rf_solution; tol=1.e-6)
        @test converged
        @test isapprox(norm(rf_solution(x)), 0.0, atol=1.e-6)
        @test drdx ≈ ForwardDiff.jacobian(rf_solution, x)
        @test isa(first(x), T)
        @test isa(first(drdx), T)

        x, drdx, converged = newtonsolve(x0, rf_nosolution; tol=1.e-6, maxiter=4)
        @test ~converged
        @test isa(first(x),T)
        @test isa(first(drdx), T)
    end
end

@testset "newtonsolve (scalar)" begin
    for T in (Float32, Float64)    
        a, x0 = rand(T,2)
        rf_solution(x) = a*x^3 - one(T)
        rf_nosolution(x) = a*x^4 + one(T)

        x, drdx, converged = newtonsolve(x0, rf_solution)
        @test converged
        @test isapprox(norm(rf_solution(x)), 0.0, atol=1.e-6)
        @test drdx ≈ ForwardDiff.derivative(rf_solution, x)
        @test isa(x, T)
        @test isa(drdx, T)

        x, drdx, converged = newtonsolve(x0, rf_nosolution)
        @test !converged 
        @test isa(x, T)
        @test isa(drdx, T)
    end
end

@testset "Multithreaded" begin
    if Threads.nthreads()==1 
        @warn("Multithreaded test should run with julia using more than one thread")
    end
    nsize = 4
    num_cases = 10
    (a,b,x0) = [rand(nsize) for _ in 1:3]
    tol = 1.e-10
    
    # Basic functionality
    function rf_solution!(r, x)
        r .= - a + b.*x + exp.(x)
    end
    rf_solution(x) = - a + b.*x + exp.(x)

    x0_v = [copy(x0) for _ in 1:num_cases]
    checks_dynamic = zeros(Bool, (3,num_cases))
    checks_static = zeros(Bool, (3,num_cases))
    
    Threads.@threads for i in 1:num_cases
        x0_d = x0_v[i]
        x0_s = SVector{nsize}(x0_d)
        # Dynamic
        x_d, drdx_d, converged_d = newtonsolve(x0_d, rf_solution!; tol=tol)
        r_check = similar(x0_d)
        checks_dynamic[1,i] = converged_d
        checks_dynamic[2,i] = isapprox(rf_solution!(r_check, x_d), zero(r_check); atol=tol)
        checks_dynamic[3,i] = (drdx_d ≈ ForwardDiff.jacobian(rf_solution!, r_check, x_d))

        # Static
        x_s, drdx_s, converged_s = newtonsolve(x0_s, rf_solution; tol=tol)
        checks_static[1,i] = converged_s
        checks_static[2,i] = isapprox(norm(rf_solution(x_s)), 0.0, atol=tol)
        checks_static[3,i] = (drdx_s ≈ ForwardDiff.jacobian(rf_solution, x_s))
    end
    @test all(checks_dynamic)
    @test all(checks_static)
end