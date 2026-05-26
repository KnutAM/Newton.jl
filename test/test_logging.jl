using Preferences

if Newton.LOGGING
    @testset "logging_mode" begin
        @test Newton.default_logger(rand(2)) isa Newton.StandardLogger
        @test Preferences.load_preference(Newton, "log_iterations")
        Newton.logging_mode(;enable=false)
        @test !(Preferences.load_preference(Newton, "log_iterations"))
        Newton.logging_mode(;enable=true) # Test that it sets even if already enabled 
        @test Preferences.load_preference(Newton, "log_iterations")
    end
else
    @testset "logging_mode" begin
        @test Newton.default_logger(rand(2)) isa Newton.NoLogger
    end
end

function capture_stdout(f)
    out = Pipe()
    redirect_stdout(out) do
        f()
    end
    close(out.in)
    return read(out, String)
end

@testset "logging" begin
    rf(x::SVector{2}) = SVector((x[1]^2 * (x[2] - 1)^2 - x[1] - 1, x[1] + x[2]))
    function rf!(r::Vector, x::Vector)
        return r .= rf(SVector((x[1], x[2])))
    end
    rf(x::Vec{2}) = Vec{2}(rf(SVector((x[1], x[2]))))
    function rf(x::SymmetricTensor{2,2})
        r = rf(SVector((x[1,1], x[2,2])))
        return SymmetricTensor{2,2}((r[1], r[2], x[2,1]))
    end
    rf(x::Tensor{2,1}) = Tensor{2,1}((rf(x[1,1]),))
    rf(x::Number) = 1 - x^2 + x
    residuals(l::Newton.StandardLogger) = l.residuals
    residuals(l::Newton.FullLogger) = norm.(l.residuals)
    last_residual(l::Newton.StandardLogger) = last(l.residuals)
    last_residual(l::Newton.FullLogger) = norm(last(l.residuals))

    for x0 in (rand(2), rand(SVector{2}), rand(Vec{2}), rand(SymmetricTensor{2,2}), rand(), rand(Tensor{2,1}))
        log_msgs = Vector{String}[]
        for Logger in (Newton.StandardLogger, Newton.FullLogger)
            logger = Logger(x0)
            if isa(x0, Vector) # Dynamic
                @show x0
                x, drdx, converged = newtonsolve(rf!, copy(x0); logger)
                @test norm(rf!(zeros(2), x)) ≈ last_residual(logger)
            else
                x, drdx, converged = newtonsolve(rf, x0; logger)
                @test norm(rf(x)) ≈ last_residual(logger)
            end
            @test converged
            r = residuals(logger)
            @test all(r[1:end-1] .!= r[2:end]) # Ensure updated and not aliased/showing the same
            logmsg = capture_stdout() do
                Newton.report_logger(logger)
            end
            for needle in ["Residual = ", "iter", "||r||", "||Δx||", "conv. rate", string(nameof(Logger))]
                @test contains(logmsg, needle)
            end
            push!(log_msgs, split(logmsg, "\n"))
        end
        for linenr in 2:length(first(log_msgs))
            for lognr in 2:length(log_msgs)
                @test log_msgs[1][linenr] == log_msgs[lognr][linenr]
            end
        end
    end
end