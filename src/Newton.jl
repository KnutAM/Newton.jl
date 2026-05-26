module Newton
using LinearAlgebra
using DiffResults
using ForwardDiff
using StaticArrays
using Printf
import Tensors: 
    Tensors, AbstractTensor, Vec, SecondOrderTensor, FourthOrderTensor, 
    Tensor, SymmetricTensor, ⊡

export newtonsolve, ad_newtonsolve
export NewtonCache
export getx

include("compat.jl") # Defines @public
@public StandardLinsolver, UnsafeFastLinsolver, RecursiveFactorizationLinsolver
@public NoLogger, StandardLogger, FullLogger

include("logging.jl")

include("NewtonCache.jl")
include("linsolve.jl")
include("newtonsolve.jl")
include("ad_solve.jl")
include("inverse.jl")

include("deprecated.jl")

end
