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

function CyclicShiftMultiplication(X::AbstractArray,A::Vector{<:AbstractMatrix},M::Vector{Int} )
    P = getPermutation(X, A,M)
    X = permutedims(X, P)
    d = length(A)
    dims = size(X)
    #X = unfold(X, 1) -> not sure if this is needed because we are already saved this way (column major order), so i think we can just do :
    X = reshape(X, dims[1], div(prod(dims),dims[1]))
    a2 = 1
    a1 = 1
    final_dims = []
    for i in 1:d
        a1 = prod(dims[i+2:length(dims)])
        a2 = size(A[i])[1] * a2 
        push!(final_dims, size(A[i],1)) #this (hopefully) adds all the final dimensions in the right order)
        X =  transpose(A[i]) * transpose(X) # we do this so we are in a strided non-adjoint matrix 
        X = reshape(X, div(size(X)[1],(a1*a2)), size(X)[2]*a1*a2)
    end
    X = reshape(X, final_dims...)
    X = permutedims(X, invperm(P)) #get back to original indexation of the tensor
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

function getPermutation(X::AbstractArray,A::Vector{<:AbstractMatrix},M::Vector{Int})
    P = OptimalOrdering(X,M)
    return P
end
