multiinput_rf!(r::Vector, x::Vector, A::Matrix, b::Vector) = (r .= b .+ A*x)

function determine_solution(A, b)
    x0 = zero(b)
    rf!(r, x) = multiinput_rf!(r, x, A, b)
    cache = NewtonCache(x0)
    x, _, _ = newtonsolve(rf!, x0, cache)
    return x
end

@testset "newtonsolve (dynamic)" begin
    nsize = 4
    (a,b,x0) = [rand(nsize) for _ in 1:3]
    tol = 1.e-10
    
    # Basic functionality
    function rf_solution!(r, x)
        r .= - a + b.*x + exp.(x)
    end
    cache = NewtonCache(x0)
    @test x0 !== getx(cache) # Check that x0 is not aliased to getx(cache)
    xguess = getx(cache)
    copy!(xguess, x0)
    x, drdx, converged = newtonsolve(rf_solution!, xguess; tol=tol)
    r_check = similar(x0)
    @test x0 == xguess     # Input should not be modified when not aliased
    @test converged
    @test isapprox(rf_solution!(r_check, x), zero(r_check); atol=tol)
    @test drdx ≈ ForwardDiff.jacobian(rf_solution!, r_check, x)

    # Test with given cache
    x1 = copy(getx(cache))
    copy!(getx(cache), x0)
    x, drdx, converged = newtonsolve(rf_solution!, getx(cache), cache; tol=tol)
    @test x === getx(cache) # Output should be aliased to cache
    x, drdx, converged = newtonsolve(rf_solution!, x1, cache; tol=tol)
    @test x === getx(cache) !== x1 # Output x should be aliased to cache, not to input

    function rf_nosolution!(r, x)
        r .= a .+ b.*x.^2
    end
    x = copy(x0)
    x, drdx, converged = newtonsolve(rf_nosolution!, x)
    @test !converged

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

        x, drdx, converged = newtonsolve(rf_solution, x0; tol=1.e-6)
        @test converged
        @test isapprox(norm(rf_solution(x)), 0.0, atol=1.e-6)
        @test drdx ≈ ForwardDiff.jacobian(rf_solution, x)
        @test isa(first(x), T)
        @test isa(first(drdx), T)

        x, drdx, converged = newtonsolve(rf_nosolution, x0; tol=1.e-6, maxiter=4)
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

        x, drdx, converged = newtonsolve(rf_solution, x0)
        @test converged
        @test isapprox(norm(rf_solution(x)), 0.0, atol=1.e-6)
        @test drdx ≈ ForwardDiff.derivative(rf_solution, x)
        @test isa(x, T)
        @test isa(drdx, T)

        x, drdx, converged = newtonsolve(rf_nosolution, x0)
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
    (a, b, x0_s) = [rand(SVector{nsize}) for _ in 1:3]
    tol = 1.e-10

    # Basic functionality
    function rf_solution!(r, x)
        r .= - a + b.*x + exp.(x)
    end
    rf_solution(x) = - a + b.*x + exp.(x)

    checks_dynamic = zeros(Bool, (3,num_cases))
    checks_static = zeros(Bool, (3,num_cases))
    
    Threads.@threads for i in 1:num_cases
        x0_d = Vector(x0_s)
        # Dynamic
        x_d, drdx_d, converged_d = newtonsolve(rf_solution!, x0_d; tol=tol)
        r_check = similar(x0_d)
        checks_dynamic[1,i] = converged_d
        checks_dynamic[2,i] = isapprox(rf_solution!(r_check, x_d), zero(r_check); atol=tol)
        checks_dynamic[3,i] = (drdx_d ≈ ForwardDiff.jacobian(rf_solution!, r_check, x_d))

        # Static
        x_s, drdx_s, converged_s = newtonsolve(rf_solution, x0_s; tol=tol)
        checks_static[1,i] = converged_s
        checks_static[2,i] = isapprox(norm(rf_solution(x_s)), 0.0, atol=tol)
        checks_static[3,i] = (drdx_s ≈ ForwardDiff.jacobian(rf_solution, x_s))
    end
    @test all(checks_dynamic)
    @test all(checks_static)
end
