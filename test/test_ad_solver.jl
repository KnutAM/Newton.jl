
function inner_residual(x::Tensors.Vec{2}, a::Tensors.Tensor{2,2}, b::SVector{2})
    return Tensors.Vec((
        x ⋅ x - b[1] * a ⊡ a * x[2],
        b[2] * x ⋅ a ⋅ x - b[1]*b[2] * x[1]
    ))
end
function outer_function!(rr::Vector, aa::Vector)
    @assert length(rr) == length(aa) == 6
    x0 = zero(Tensors.Vec{2})
    a = Tensors.Tensor{2,2}(tuple(aa[1:4]...))
    b = SVector((aa[5], aa[6]))
    x, converged = ad_newtonsolve(inner_residual, x0, (a, b))
    @assert converged 
    rr[1:2] .= x
    rr[3:4] .= b
    rr[5:6] .= (x .* b)
    return rr 
end

@testset "ad_solver" begin
    aa = rand(6)
    ac = copy(aa)
    rr = zeros(6); rplus = zeros(6); rminus = zeros(6)
    drda_ad = ForwardDiff.jacobian(outer_function!, rr, aa)
    drda_num = fill!(similar(drda_ad), NaN)
    @assert isnan(drda_num[1,1])
    FiniteDiff.finite_difference_jacobian!(drda_num, outer_function!, aa)
    @test drda_ad ≈ drda_num
end
