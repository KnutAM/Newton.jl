import Preferences

# logging setup modified from Ferrite.debug_mode. 
const LOGGING = Preferences.@load_preference("log_iterations", false)

"""
    Newton.debug_mode(; enable=true)
Helper to turn on (`enable=true`) or off (`enable=false`) logging of iterations in `Newton.jl`.
Internally, changes the how `Newton.@if_logging expr` is evaluated: 
when logging mode is enabled, `expr` is evaluated, otherwise `expr` is ignored.
"""
function logging_mode(; enable = true)
    if LOGGING == enable == true
        @info "Logging mode already enabled."
    elseif LOGGING == enable == false
        @info "Logging mode already disabled."
    else
        Preferences.@set_preferences!("log_iterations" => enable)
        @info "Logging mode $(enable ? "en" : "dis")abled. Restart the Julia session for this change to take effect!"
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

function show_iteration_trace(errs::Vector)
    println("Did not convergen in $(length(errs)) iteration, residual = $(last(errs))")
    # Rate of convergence, q: |rₖ₋₁|/|rₖ₋₂|^q = μ
    # Hence, q log(|rₖ₋₂|) + log(μ) = log(|rₖ₋₁|)
    # And as well, q log(|rₖ₋₁|) + log(μ) = log(|rₖ|)
    # Such that q = log(|rₖ₋₁|/|rₖ|)/log(|rₖ₋₂|/|rₖ₋₁|)
    for k in eachindex(errs)
        print("$k: $(errs[i])")
        if k>=3
            q = log(errs[k-1]/errs[k])/log(errs[k-2]/errs[k-1])
            println(" q=$q")
        else
            println()
        end
    end
end
