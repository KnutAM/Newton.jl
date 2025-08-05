module Newton
using LinearAlgebra
using DiffResults
using ForwardDiff
using StaticArrays
using Printf
import Tensors: 
    Tensors, AbstractTensor, Vec, SecondOrderTensor, FourthOrderTensor, 
    ⊡

export newtonsolve, ad_newtonsolve
export NewtonCache
export getx

public StandardLinsolver, UnsafeFastLinsolver, RecursiveFactorizationLinsolver

include("utils.jl")

include("NewtonCache.jl")
include("linsolve.jl")
include("newtonsolve.jl")
include("ad_solve.jl")
include("inverse.jl")

include("deprecated.jl")

end
