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
    P = M
    X = permutedims(X, P)
    d = length(A)
    dims = collect(size(X)) #use collect so we can change final_dims 
    #X = unfold(X, 1) -> not sure if this is needed because we are already saved this way (column major order), so i think we can just do :
    X = reshape(X, dims[1], :)
    first_dims = size.(A, 1)
    a2 = 1
    first_dims = insert!(first_dims, 1, 1)
    final_dims = dims
    for i in 1:d
        a2 = first_dims[i] * a2
        #blijkbaar is deze fout
        setindex!(final_dims, size(A[i], 1), i)
        # switch deze volgorde  
        X = transpose(X) * transpose(A[i])   # we do this so we are in a strided non-adjoint matrix 
        #reshape met : gebruiken
        X = reshape(X, div(size(X)[1], (prod(dims[i+2:length(dims)]) * a2)), :)
        println(X)
        println(size(X))
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

function getPermutation(X::AbstractArray, A::Vector{<:AbstractMatrix}, M::Vector{Int})
    P = OptimalOrdering(X, M)
    return P
end