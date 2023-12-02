@testset "inv" begin
    for n in (3, 10, 20, 30, 100)
        A = rand(n,n) + n*LinearAlgebra.I
        cache = NewtonCache(A[:,1], identity)
        Ainv = inv(A)
        Ainv2 = Newton.inv!(A, cache)
        @test Ainv ≈ Ainv2
    end
end