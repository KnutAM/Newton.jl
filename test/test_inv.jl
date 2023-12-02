@testset "inv" begin
    for T in (Float32, Float64)    
        for n in (3, 10, 20, 30, 100)
            A = rand(T, n, n) + n * LinearAlgebra.I
            cache = NewtonCache(A[:,1], identity)
            Ainv = inv(A)
            Ainv2 = Newton.inv!(A, cache)
            @test Ainv ≈ Ainv2
            if VERSION ≥ v"1.9" # Functionality only active on julia 1.9 and later
                lwork = length(cache.blas_work)
                if T===Float64 # Test that it has resized correctly
                    @test lwork ≥ n
                end
                # Check that blas_work doesn't resize when called multiple times
                Newton.inv!(A, cache)
                @test lwork == length(cache.blas_work) 
            end
        end
    end
end
