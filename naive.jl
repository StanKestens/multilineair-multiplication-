using TensorToolbox
using LinearAlgebra

#X is a tensor
#A contains the matrices to multiply with
#modes contains the modes to multiply along

function naive_multiplication(X::AbstractArray,A::MatrixCell)
    sz = size(X)
    for i in 1:length(A)
        X_unfolded = tenmat(X, i)
        X_multiplied = A[i] * X_unfolded
        sz[A[i]] = size(A[i], 1)
        X = matten(X_multiplied, i, sz)
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
Y_naive = naive_multiplication(X,A);
println(size(Y_naive))  # Expected output: (2, 2, 2, 2)

function fold()

end