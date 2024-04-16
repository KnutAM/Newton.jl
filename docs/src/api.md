```@meta
CurrentModule = Newton
```

## API
### Standard usage
```@docs
newtonsolve
NewtonCache
getx
Newton.logging_mode
```

### Fast inverse
```@docs
Newton.inv!
```

### Use inside AD-calls
```@docs
ad_newtonsolve
```

This approach is faster then naively differentiating a call which includes a newtonsolve,
as we avoid iterating using `Dual` numbers. 
```julia
using Newton, Tensors, ForwardDiff, BenchmarkTools
rf(x::Vec, a::Number) = a * x - (x ⋅ x) * x
function myfun!(outputs::Vector, inputs::Vector)
    x0 = ones(Vec{2}) # Initial guess
    a = inputs[1] + 2 * inputs[2]
    x, converged = ad_newtonsolve(rf, x0, (a,))
    outputs[1] = x ⋅ x
    outputs[2] = a * x[1]
    return outputs 
end
function myfun2!(outputs::Vector, inputs::Vector)
    x0 = ones(Vec{2}) # Initial guess
    a = inputs[1] + 2 * inputs[2]
    x, _, converged = newtonsolve(x -> rf(x, a), x0)
    outputs[1] = x ⋅ x
    outputs[2] = a * x[1]
    return outputs
end
J = zeros(2,2)
out = zeros(2); inp = [1.2, 0.5]
cfg = ForwardDiff.JacobianConfig(myfun!, out, inp)
cfg2 = ForwardDiff.JacobianConfig(myfun2!, out, inp)
@btime ForwardDiff.jacobian!($J, $myfun2!, $out, $inp, $cfg2);  # 285.662 ns (0 allocations: 0 bytes)
@btime myfun2!($out, $inp);                                     # 143.381 ns (0 allocations: 0 bytes)
@btime ForwardDiff.jacobian!($J, $myfun!, $out, $inp, $cfg);    # 183.359 ns (0 allocations: 0 bytes)
@btime myfun!($out, $inp);                                      # 143.381 ns (0 allocations: 0 bytes)
```
showing that we get quite close to a regular non-differentiating call wrt. computational time in this microbenchmark.

## Internal API
```@docs
Newton.linsolve!
```
