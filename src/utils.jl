import Preferences

# logging setup modified from Ferrite.debug_mode. 
const LOGGING = Preferences.@load_preference("log_iterations", false)

"""
    Newton.logging_mode(; enable=true)
Helper to turn on (`enable=true`) or off (`enable=false`) logging of iterations in `Newton.jl`.
Internally, changes the how `Newton.@if_logging expr` is evaluated: 
when logging mode is enabled, `expr` is evaluated, otherwise `expr` is ignored.
"""
function logging_mode(; enable = true)
    Preferences.@set_preferences!("log_iterations" => enable)
    @info "Logging mode $(enable ? "en" : "dis")abled."
    if LOGGING == enable
        @info "Logging mode did not change though"
    else
        @info "Logging mode changed: Restart the Julia session for this change to take effect!"
    end
end

@static if LOGGING
    @eval begin
        macro if_logging(ex)
            return :($(esc(ex)))
        end
    end
else
     @eval begin
        macro if_logging(ex)
            return nothing
        end
    end
end

function show_iteration_trace(errs::Vector, resids, tol)
    niter = length(errs)
    println("Did not converge in $niter iterations")
    println("Residual = $(last(errs)), tolerance = $tol")
    if !isnothing(resids)
        for k in unique((1,min(2,niter),niter))
            println("Each residual entry at iteration $k/$niter, |r|=$(norm(resids[k])) / $(errs[k])")
            for (i, r) in enumerate(resids[k])
                @printf("%2.0f: %10.3e\n", i, r)
            end
        end
    end
    # Rate of convergence, q: |rₖ₋₁|/|rₖ₋₂|^q = μ
    # Hence, q log(|rₖ₋₂|) + log(μ) = log(|rₖ₋₁|)
    # And as well, q log(|rₖ₋₁|) + log(μ) = log(|rₖ|)
    # Such that q = log(|rₖ₋₁|/|rₖ|)/log(|rₖ₋₂|/|rₖ₋₁|)
    println("iter: |r|, [convergence rate]")
    for k in eachindex(errs)
        @printf("%4.0f: %10.3e", k, errs[k])
        if k>=3
            q = log(errs[k-1]/errs[k])/log(errs[k-2]/errs[k-1])
            @printf(", %4.2f", q)
        end
        println()
    end
end
