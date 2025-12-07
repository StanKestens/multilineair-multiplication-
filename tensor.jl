using TensorToolbox
using Test
using LinearAlgebra
"""
Function to unfold a tensor into a matrix in mode-n 
Input: X: tensor (AbstractArray)
        mode: integer specifying the mode along which to unfold
Output: unfolded tensor as a matrix in mode n
"""
function unfold(X, n::Integer)
    N  = ndims(X)
    sz = size(X)
    p  = (n, setdiff(1:N, n)...)

    # Geen echte permute-kopie: alleen een view met andere indexmapping
    Yp = PermutedDimsArray(X, p)

    # reshape maakt ook geen kopie
    return reshape(Yp, sz[n], :)
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