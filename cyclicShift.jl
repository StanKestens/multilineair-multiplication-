using TensorToolbox
using LinearAlgebra

include("tensor.jl")
include("ordering.jl")

"""
In this file we attempt to implement cyclic shifts, this is an alternative way to do multilineair multiplication, 
which only uses a single transposition, this is mostly based on the psuedocode given in the W. Baert and N.Vannieuwenhoven paper on ATC.
Do note that this does not use the optimal ordering. 
This first implementation only works when given d matrices with d being the order of X

Input : 
    -X, a tensor
    -A , a collection of matrices
    -M , the permutation were gonna have to do to get to the correct order 
Output : X, the same tensor multiplied by each each matrix in A

"""

function CyclicShiftMultiplication(X::AbstractArray, A::Vector{<:AbstractMatrix}, M::Vector{Int})
    m = length(A)
    dims = collect(size(X))
    X = reshape(X, dims[1], :)
    R = size.(A, 1) #list of new dimensions 
    R = insert!(R, 1, 1)
    a = 1
    for i in 1:m
        a = R[i] * a
        # switch deze volgorde  
        X = transpose(X) * transpose(A[i])   # we do this so we are in a strided non-adjoint matrix 
        #reshape met : gebruiken
        X = reshape(X, div(size(X)[1], (prod(dims[i+2:length(dims)]) * a)), :)
    end
    R = deleteat!(R, 1)
    X = reshape(X, vcat(R, dims[m+1:end])...)
    return X
end

"""
This function will decide the optimal way to calculate the cyclic shift based on :
    Input:
    -Tensor X 
    -Vector of Matrices A 
    -Modes of multiplication M
    Output:
    -Permutation P how we should permute our tensor so the cyclic shift works optimal

    1ste versie : werkt enkel als |M| = order(A)
"""

function getPermutation(X::AbstractArray, A::Vector{<:AbstractMatrix}, M::Vector{Int})
    P = OptimalOrdering(X, M)
    return P
end