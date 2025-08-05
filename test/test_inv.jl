@testset "inv!" begin
    for (linsolver, ns) in ((Newton.StandardLinsolver(), (3, 10, 30, 100)), (Newton.RecursiveFactorizationLinsolver(), 10),)# (Newton.UnsafeFastLinsolver(), 10))
        for n in ns
            for T in (n == 10 ? (Float32, Float64) : (Float64,))
                A = rand(T, n, n) + n * LinearAlgebra.I
                cache = NewtonCache(A[:,1]; linsolver)
                Ainv = inv(A)
                Ainv2 = Newton.inv!(A, cache)
                @test Ainv ≈ Ainv2
                 # Functionality only active on julia 1.9 and later
                if VERSION ≥ v"1.9" && isa(linsolver, Newton.StandardLinsolver)
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
end

@testset "sinv" begin
    for n in [4, 10, 40]
        A = SMatrix{n, n}(rand(n, n)) + n * LinearAlgebra.I
        @test inv(A) ≈ Newton.sinv(A)
    end
end
