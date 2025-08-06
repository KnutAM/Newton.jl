@testset "linsolve!" begin
    @testset for linsolver in (
        Newton.StandardLinsolver(), 
        #VERSION ≥ v"1.11" ? Newton.RecursiveFactorizationLinsolver() : nothing, 
        Newton.UnsafeFastLinsolver()
        )
        linsolver === nothing && continue
        for n in (10, 25)
            A = 2*I + rand(10,10)
            b = rand(10)
            A1, A2 = [copy(A) for _ in 1:2]
            b1, b2 = [copy(b) for _ in 1:2]
            @test A1\b1 ≈ Newton.linsolve!(A2, b2, NewtonCache(b; linsolver))
        end
    end
end

@testset "linsolve" begin
    @testset for linsolver in (Newton.StandardLinsolver(), Newton.UnsafeFastLinsolver())
        @testset "scalar" begin
            K, x = rand(2)
            b = K * x
            @test x ≈ Newton.linsolve(linsolver, K, b)
        end
        @testset "SMatrix" begin
            K, x = (rand(SMatrix{2, 2}) + LinearAlgebra.I, rand(SVector{2}))
            b = K * x
            @test x ≈ Newton.linsolve(linsolver, K, b)
        end
        @testset "Tensors" begin
            @testset for (op, order) in ((⋅, 2), (⊡, 4))
                @testset for T in (Tensor, SymmetricTensor)
                    K, x = (rand(T{order, 3}) + one(Tensor{order, 3}), rand(Tensor{order÷2, 3}))
                    b = op(K, x)
                    @test x ≈ Newton.linsolve(linsolver, K, b)
                end
            end
        end
    end
end
