using TensorToolbox
using Test
using BenchmarkTools
using LinearAlgebra
"""
Function to unfold a tensor into a matrix in mode-n 
Input: X: tensor (AbstractArray)
        mode: integer specifying the mode along which to unfold
Output: unfolded tensor as a matrix in mode n
"""
function unfold(X::AbstractArray, n::Integer)
    N = ndims(X)
    sz = size(X)
    # Bring mode n to the first dimension, keep relative order of others
    perm = (n, (1:N)[(1:N) .!= n]...)
    Xp = permutedims(X, perm)
    reshape(Xp, sz[n], :)
end

"""
Function to fold an matrix into a tensor
Input:
    A - The matrix
    dim - vector with the dimensions of the final tensor
    n-mode in which were folding
Output:
    X - Tensor , the fold of A

"""

function matten(A::AbstractMatrix, n::Integer,dim::Vector{Int})
  m = setdiff(1:length(dim), n)
  X = reshape(A,[dim[n];dim[m]]...)
  permutedims(X,invperm([n;m]))
end
