@testset "deprecated" begin
    @testset "argument order" begin
        rf!(r, x) = (r .= x)
        rf(x) = x 
        for x0 in (rand(3), rand(SVector{3}), rand())
            f = isa(x0, Vector) ? rf! : rf     
            x, drdx, converged = newtonsolve(f, x0) # Correct syntax 
            @test converged
            x_dep, drdx_dep, converged_dep = newtonsolve(x0, f) # Deprecated syntax 
            @test converged_dep
            @test x ≈ x_dep 
            @test drdx ≈ drdx_dep
        end
    end
end
