macro public(ex)
    if VERSION >= v"1.11.0-DEV.469"
        args = ex isa Symbol ? (ex,) : Base.isexpr(ex, :tuple) ? ex.args : error("something informative")
        esc(Expr(:public, args...))
    else
        nothing
    end
end