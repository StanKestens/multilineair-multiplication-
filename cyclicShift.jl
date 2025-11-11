using TensorToolbox
using LinearAlgebra

include("tensor.jl")

"""
In this file we attempt to implement cyclic shifts, this is an alternative way to do multilineair multiplication, 
which only uses a single transposition, this is mostly based on the psuedocode given in the W. Baert and N.Vannieuwenhoven paper on ATC.
Do note that this does not use the optimal ordering. 
This first implementation only works when given d matrices with d being the order of X

Input : 
    -X, a tensor
    -A , a collection of matrices
Output : X, the same tensor multiplied by each each matrix in A

"""
function CyclicShiftMultiplication(X::AbstractArray,A::MatrixCell)
    d=length(A)

    for i in 1:d#overlopen dimensies
        B = reshape(permutedims(X, (i, setdiff(1:d, i)...)), size(X, i), :)
        B = A[i] * B
        newdims = (size(Ulist[i], 1), size(X)[setdiff(1:d, i)]...)
        X = permutedims(reshape(A, newdims), invperm((i, setdiff(1:d, i)...)))
    end
    return X
end 