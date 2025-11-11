using TensorToolbox
using Test
using BenchmarkTools
using LinearAlgebra

# input: X: tensor (AbstractArray)
#        mode: integer specifying the mode along which to unfold
# output: unfolded tensor as a matrix in mode n

function unfold(X::AbstractArray, n::Integer)
    N = ndims(X)
    sz = size(X)
    # Bring mode n to the first dimension, keep relative order of others
    perm = (n, (1:N)[(1:N) .!= n]...)
    Xp = permutedims(X, perm)
    reshape(Xp, sz[n], :)
end