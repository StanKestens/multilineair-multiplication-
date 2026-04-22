using TensorToolbox
using LinearAlgebra
include("ordering.jl")
include("../utility/tensor.jl")
"""
Input: X is a tensor
    A contains the matrices to multiply with
    modes contains the modes to multiply along
Output : 
"""

function NonNaiveMultiplication(X::AbstractArray, A::Vector{<:AbstractMatrix})
    sz = size(X)
    order = OptimalOrdering(X, A)
    #eerst permute voor order
    for i in order
        X_unfolded = unfold(X, i)
        X_multiplied = A[i] * X_unfolded
        new_sz = collect(sz)
        new_sz[i] = size(A[i], 1)
        X = fold(X_multiplied, i, new_sz)
        sz = size(X)
    end
    return X
end# Expected output: (5, 4, 3)