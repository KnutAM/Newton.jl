```@meta
CurrentModule = Newton
```

# Newton

Documentation for [Newton](https://github.com/KnutAM/Newton.jl).

```@index
```

## Basic usage
Define a mutating residual function 
```julia
nsize=4
(a,b) = [rand(nsize) for _ in 1:2]
function rf!(r, x)
    r .= - a + b.*x + exp.(x)
end
```

Define initial guess
```julia
x=zeros(nsize)
```
Preallocate cache 
```julia
cache = NewtonCache(x,rf!)
```

Solve the problem
```julia
drdx = get_drdx(cache)  # Can also do drdx=zeros(nsize,nsize), 
                        # but this would allocate unecessary
converged = newtonsolve!(x, drdx, rf!, cache)
```



```@autodocs
Modules = [Newton]
```
