
function inner_residual(x::Vec{2}, a::Tensor{2,2}, b::SVector{2}, c=1.0, d=ones(Vec{2}), e=2.0, f=1.0)
    return Vec{2}((
        x ⋅ x - b[1] * a ⊡ a * x[2] * c * f,
        b[2] * x ⋅ a ⋅ x - b[1]*b[2] * x[1] * norm(d) * e,
    ))
end
function outer_function2!(rr::Vector, aa::Vector)
    @assert length(rr) == length(aa) == 6
    x0 = ones(Vec{2})
    a = Tensor{2,2}(ntuple(i -> aa[i], 4))
    b = SVector((aa[5], aa[6]))
    x, converged = ad_newtonsolve(inner_residual, x0, (a, b); tol=1e-12)
    @assert converged 
    rr[1:2] .= x
    rr[3:4] .= b
    rr[5:6] .= (x .* b)
    return rr 
end
function outer_function6!(rr::Vector, aa::Vector)
    @assert length(rr) == length(aa) == 6
    x0 = ones(Vec{2})
    a = Tensor{2,2}(ntuple(i -> aa[i], 4))
    b = SVector((aa[5], aa[6]))
    c, e, f = aa[1:3]
    d = Vec((aa[1], aa[4]))
    x, converged = ad_newtonsolve(inner_residual, x0, (a, b, c, d, e, f); tol=1e-12)
    @assert converged
    rr[1:2] .= x
    rr[3:4] .= b
    rr[5:6] .= (x .* b)
    return rr
end

@testset "ad_solver" begin
    aa = [0.4, 0.8, 0.5, 0.5, 0.9, 0.4] # Some vector for which there exists a solution
    rr = zeros(6)
    for f in (outer_function2!, outer_function6!)
        drda_ad = ForwardDiff.jacobian(f, rr, aa)
        drda_num = fill!(similar(drda_ad), NaN)
        @assert isnan(drda_num[1,1])
        FiniteDiff.finite_difference_jacobian!(drda_num, f, aa)
        @test isapprox(drda_ad, drda_num; atol = 1e-6)
    end
end
