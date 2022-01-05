using Newton
using Test
using ForwardDiff
using LinearAlgebra

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
        
end
