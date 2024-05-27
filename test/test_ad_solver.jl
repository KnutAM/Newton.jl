
function inner_residual(x::Vec{2}, a::Tensor{2,2}, b::SVector{2}, c=1.0, d=ones(Vec{2}), e=2.0, f=1.0)
    return Vec{2}((
        x ⋅ x - b[1] * a ⊡ a * x[2] * c * f,
        b[2] * x ⋅ a ⋅ x - b[1]*b[2] * x[1] * norm(d) * e,
    ))
end
function inner_residual(x::SVector{2}, args::Number...)
    return SVector((x[1] - sum(args), x[2] + prod(args)))
end
function inner_residual(x::SymmetricTensor{2,2}, args::Vec...)
    s = sum(norm, args)
    p = prod(norm, args) + one(s)
    return SymmetricTensor{2,2}((x[1,1] + s, x[2,1], x[2,2] + x[2,2]*x[1,1] - p))
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
function outer_functionN_svec!(rr::Vector, aa::Vector, N::Int, )
    @assert length(rr) == length(aa) == 2
    args = ntuple(i -> aa[1]^i - i * aa[2], N)
    x0 = SVector((0.0, 0.0))
    x, converged = ad_newtonsolve(inner_residual, x0, args; tol=1e-12)
    @assert converged
    rr[1] = x[2]^2 * aa[1]*aa[2]
    rr[2] = x[1]/(aa[2] + 1)
    return rr
end

function outer_functionN_tensor!(rr::Vector, aa::Vector, N::Int, )
    @assert length(rr) == length(aa) == 2
    args = ntuple(i -> Vec((aa[1]^i, aa[2]/i)), N)
    x0 = zero(SymmetricTensor{2,2})
    x, converged = ad_newtonsolve(inner_residual, x0, args; tol=1e-12)
    @assert converged 
    rr[1] = norm(x)^2 * aa[1]
    rr[2] = tr(x)^2 / aa[2]
    return rr
end

@testset "ad_solver" begin
    for f in (outer_function2!, outer_function6!)
        @testset "$f" begin
            aa = [0.4, 0.8, 0.5, 0.5, 0.9, 0.4] # Some vector for which there exists a solution
            rr = fill(NaN, length(aa))
            rc = fill(NaN, length(aa))
            drda_ad = ForwardDiff.jacobian(f, rr, aa)
            f(rc, aa)
            @test rc ≈ rr
            drda_num = fill!(similar(drda_ad), NaN)
            @assert isnan(drda_num[1,1])
            FiniteDiff.finite_difference_jacobian!(drda_num, f, aa)
            @test isapprox(drda_ad, drda_num; atol = 1e-6, rtol = 1e-6)
        end
    end
    for f in (outer_functionN_svec!, outer_functionN_tensor!)
        @testset "$f" begin
            for N in 1:6
                @testset "N = $N" begin
                    aa = [0.25, 0.3]
                    rr = fill(NaN, length(aa))
                    rc = fill(NaN, length(aa))
                    g(r, a) = f(r, a, N)
                    drda_ad = ForwardDiff.jacobian(g, rr, aa)
                    g(rc, aa)
                    @test rc ≈ rr
                    drda_num = fill!(similar(drda_ad), NaN)
                    @assert isnan(drda_num[1,1])
                    FiniteDiff.finite_difference_jacobian!(drda_num, g, aa)
                    @test isapprox(drda_ad, drda_num; atol = 1e-6, rtol = 1e-6)
                end
            end
        end
    end
end
