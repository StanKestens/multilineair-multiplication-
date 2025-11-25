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
function unfold(X::AbstractArray, k::Integer)
        dims = size(X)
        X = reshape(X, size(X,1), :)  # mode-1 unfolding
        for i in 2:k-1
                X = transpose(X)
                n_last = size(X, 2)
                n_now = dims[i]
                X = reshape(X, n_now, size(X,1) * n_last ÷ n_now)
        end
        return X
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

function fold(A::AbstractMatrix, n::Integer,dim::Vector{Int})
  m = setdiff(1:length(dim), n)
  X = reshape(A,[dim[n];dim[m]]...)
  permutedims(X,invperm([n;m]))
end
