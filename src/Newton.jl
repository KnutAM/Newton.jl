module Newton
using LinearAlgebra
using RecursiveFactorization
using DiffResults
using ForwardDiff
using StaticArrays
using Printf
import Tensors: Tensors, AbstractTensor

export newtonsolve, ad_newtonsolve
export NewtonCache
export getx

include("utils.jl")

include("NewtonCache.jl")
include("linsolve.jl")
include("newtonsolve.jl")
include("ad_solve.jl")
include("inverse.jl")

include("deprecated.jl")

end
