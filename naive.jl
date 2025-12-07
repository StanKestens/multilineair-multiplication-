using TensorToolbox
using LinearAlgebra

include("tensor.jl")

"""
Input: X is a tensor
    A contains the matrices to multiply with
Output : 
"""
function NaiveMultiplication(X::AbstractArray, A::MatrixCell, order::Vector{Int}) # mode order als argument meegeven
    N = ndims(X)
    P = order
    X = permutedims(X, P)
    for i in 1:N
        #@assert size(A[i], 2) == size(X, i) "A[$i] has incompatible dimensions"
        Xmat = unfold(X, i)
        Xmat = A[i] * Xmat
        sz = collect(size(X))
        sz[i] = size(A[i], 1)
        X = matten(Xmat, i, sz)
    end
    X = permutedims(X, invperm(P))
    return X
end

# Test
# X = randn(4, 5, 6)
# A = MatrixCell([randn(3,4), randn(7,5), randn(2,6)])

# Y = naive_ttm(X, A)