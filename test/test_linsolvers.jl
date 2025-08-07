@testset "linsolve!" begin
    @testset for linsolver in (
        Newton.StandardLinsolver(), 
        #VERSION ≥ v"1.11" ? Newton.RecursiveFactorizationLinsolver() : nothing, 
        Newton.UnsafeFastLinsolver()
        )
        linsolver === nothing && continue
        for n in (10, 25)
            A = 2*I + rand(n, n)
            b = rand(n)
            Ac = copy(A)
            bc = copy(b)
            @test Ac\bc ≈ Newton.linsolve!(A, b, NewtonCache(b; linsolver))
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
