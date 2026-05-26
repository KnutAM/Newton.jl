# Mechanism to enable logging via Preferences
import Preferences

# logging setup modified from Ferrite.debug_mode. 
const LOGGING = Preferences.@load_preference("log_iterations", false)

"""
    Newton.logging_mode(; enable=true)
Helper to turn on (`enable=true`) or off (`enable=false`) logging of iterations in `Newton.jl`.
This changes the default logger from [`NoLogger`](@ref) to [`StandardLogger`](@ref).
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
# End of preferences logging mode interface (just default will be defined after types introduced below)

# Some useful utils for logging
unalias(x) = copy(x)
unalias(x::AbstractTensor) = x

infer_jacobian_type(::T) where {T <: Number} = T
infer_jacobian_type(::Vector{T}) where {T} = Matrix{T}
infer_jacobian_type(::SVector{dim, T}) where {dim, T} = SMatrix{dim, dim, T}
infer_jacobian_type(x::AbstractVector) = typeof(x * x') # General for vectors
infer_jacobian_type(::Vec{dim, T}) where {dim, T} = Tensor{2, dim, T, dim^2}
infer_jacobian_type(::Tensor{2, dim, T}) where {dim, T} = Tensor{4, dim, T, dim^4}
infer_jacobian_type(::SymmetricTensor{2, dim, T, M}) where {dim, T, M} = SymmetricTensor{4, dim, T, M^2}

abstract type AbstractLogger end

"""
    reset_logger!(logger::AbstractLogger)

Reset the `logger` at the beginning of the newton iterations.
"""
function reset_logger! end

"""
    update_logger!(logger::AbstractLogger, r_norm, r, x, drdx)

Update the logger for the current iteration by passing the residual norm, `r_norm`,
the full residual, `r`, the unknowns, `x`, and the jacobian, `drdx`.
"""
function update_logger! end

"""
    report_logger(logger::AbstractLogger)

Report (print) the information from the logger.
"""
function report_logger end

"""
    NoLogger()

No-op logger that doesn't do any logging. This is used by default
if no logger is passed, and logging is not enabled via Preferences.jl
(see [Newton.logging_mode](@ref)).
"""
struct NoLogger <: AbstractLogger end

reset_logger!(::NoLogger) = nothing
update_logger!(::NoLogger, args...) = nothing
report_logger(::NoLogger) = nothing

"""
    StandardLogger(x)

The standard logger to show the evolution of residual norms and changes,
measured by norm, of the unknowns between iterations.
"""
mutable struct StandardLogger{T, XT}
    iter::Int
    const residuals::Vector{T}  # norm(r)
    const updates::Vector{T}    # norm(Δx)
    x_old::XT
end

StandardLogger(x) = StandardLogger(-1, eltype(x)[], eltype(x)[], unalias(x))

function reset_logger!(l::StandardLogger)
    l.iter = 0
    empty!(l.residuals)
    empty!(l.updates)
    return l
end

function update_logger!(l::StandardLogger, r_norm, _, x, _)
    l.iter += 1
    push!(l.residuals, r_norm)
    l.iter > 1 && push!(l.updates, norm(x - l.x_old)) # Only if not first
    l.x_old = unalias(x)
    return l
end

function report_logger(l::StandardLogger)
    println("StandardLogger: Residual = ", last(l.residuals), " after ", l.iter, " iterations")
    print_convergence_trace(l.iter, l.residuals, l.updates)
end

function print_convergence_trace(niter, residuals, updates)
    @printf("%5s | %10s | %10s | %10s |\n", "iter", "||r||", "||Δx||", "conv. rate")
    @printf("%5u | %10.3e | %10s | %10s |\n", 1, residuals[1], "", "")
    for i = 2:niter
        @printf("%5u | %10.3e | %10.3e | ", i, residuals[i], updates[i - 1])
        if i ≥ 3
            q = log(residuals[i - 1]/residuals[i])/log(residuals[i - 2]/residuals[i - 1])
            @printf("%10.3f |\n", q)
        else
            @printf("%10s |\n", "")
        end
    end
end


"""
    FullLogger(x)

A logger that logs everything known to the newton iterations,
* The full residual in each iteration
* The full unknowns in each iteration
* The full jacobian in each iteration

The [`report_logger`](@ref) will still just print the same info as the `StandardLogger`,
but additional information can be obtained by directly accessing the field. This is intended 
for interactive debugging, and the fields may change.
"""
mutable struct FullLogger{XT, KT}
    iter::Int
    const residuals::Vector{XT}
    const unknowns::Vector{XT}
    const jacobians::Vector{KT}
end

function FullLogger(x::XT) where {XT}
    KT = infer_jacobian_type(x)
    return FullLogger(-1, XT[], XT[], KT[])
end

function reset_logger!(l::FullLogger)
    l.iter = 0
    empty!(l.residuals)
    empty!(l.unknowns)
    empty!(l.jacobians)
    return l
end

function update_logger!(l::FullLogger, _, r, x, drdx)
    l.iter += 1
    push!(l.residuals, unalias(r))
    push!(l.unknowns, unalias(x))
    push!(l.jacobians, unalias(drdx))
    return l
end

function report_logger(l::FullLogger)
    println("FullLogger: Residual = ", norm(last(l.residuals)), " after ", l.iter, " iterations")
    print_convergence_trace(
        l.iter, 
        norm.(l.residuals), 
        [norm(l.unknowns[k] - l.unknowns[k - 1]) for k in 2:l.iter])
end

# Default logger
@static if LOGGING
    default_logger(x) = StandardLogger(x)
else
    default_logger(args...) = NoLogger()
end
