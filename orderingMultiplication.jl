using TensorToolbox
using LinearAlgebra
include("ordering.jl")
include("tensor.jl")
"""
Input: X is a tensor
    A contains the matrices to multiply with
    modes contains the modes to multiply along
Output : 
"""

function NonNaiveMultiplication(X::AbstractArray, A::MatrixCell)
    sz = size(X)
    order = OptimalOrdering(X,A)
    #eerst permute voor order
    for i in order
        X_unfolded = unfold(X, i)
        X_multiplied = A[i] * X_unfolded
        new_sz = collect(sz)
        new_sz[i] = size(A[i], 1)
        X = matten(X_multiplied, i, new_sz)
        sz = size(X)
    end
    return X
end

#Test
X = rand(3,4,5)
A = MatrixCell([
    rand(5,3),   # for mode 1
    rand(4,4),   # for mode 2
    rand(3,5)    # for mode 3
])

Y = NonNaiveMultiplication(X,A);
println(size(Y))  # Expected output: (2, 2, 2, 2)

function fold()

end