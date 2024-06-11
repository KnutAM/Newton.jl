@testset "AD inside residual" begin
    # Check that we don't get pertubation confusion.
    function myfun(a::SymmetricTensor{2,3,<:Tensors.Dual}, b)
        return dev(a) * b
    end
    function residual!(r::Vector, x::Vector, ::Val{UseAD}) where UseAD
        @assert length(r) == length(x) == 6
        a = frommandel(SymmetricTensor{2,3}, x)
        v = if UseAD
            gradient(z -> myfun(z, a[1,1]), a)
        else
            I2 = one(a)
            I4 = one(SymmetricTensor{4,3})
            a[1,1] * (I4 - I2 ⊗ (I2 / 3))
        end
        tomandel!(r, a + v ⊡ a - one(a))
    end

    rv = zeros(6)
    x0 = zeros(6)
    x1, drdx1, converged1 = newtonsolve((r, x) -> residual!(r, x, Val(false)), x0)
    @test converged1

    x2, drdx2, converged2 = newtonsolve((r, x) -> residual!(r, x, Val(true)), x0)
    @test converged2
    @test x1 ≈ x2 
    @test drdx1 ≈ drdx2
end