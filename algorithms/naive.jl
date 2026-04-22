using TensorToolbox
using LinearAlgebra

include("../utility/tensor.jl")

"""
Input: X is a tensor
    A contains the matrices to multiply with
Output : 
"""
function NaiveMultiplication(X::AbstractArray, A::Vector{<:AbstractMatrix}, order::Vector{Int}) # mode order als argument meegeven
    for mode in order
        Xmat = unfold(X, mode)
        Xmat = A[mode] * Xmat
        sz = collect(size(X))
        sz[mode] = size(A[mode], 1)
        X = fold(Xmat, mode, sz)
    end
    return X
end

# Test

#X = randn(4, 5, 6)
#A = MatrixCell([randn(3, 4), randn(7, 5), randn(2, 6)])

#Y = NaiveMultiplication(X, A, [1, 2, 3])
