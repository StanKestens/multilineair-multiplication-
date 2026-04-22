using TensorToolbox
using LinearAlgebra
using Pkg
using Strided

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

function CyclicShiftMultiplication(X::AbstractArray, A::Vector{<:AbstractMatrix})
    m = length(A)
    dims = collect(size(X))
    r = size.(A, 1)

    X = reshape(X, dims[1], :)

    for i in 1:m
        X = transpose(X) * transpose(A[i])
        next = i < length(dims) ? dims[i+1] : 1
        X = reshape(X, next, :)
    end

    if m == length(dims)
        return reshape(X, r...)
    else
        X = Matrix(transpose(reshape(X, prod(dims[m+1:end]), prod(r))))
        return reshape(X, vcat(r, dims[m+1:end])...)
    end
end

"""
Strategy I : filling the gaps with unity matrices


"""
function multiply_with_unity_fill(X::AbstractArray, A::Vector{<:AbstractMatrix}, active::Vector{Int})
    d = ndims(X)

    A_full = Vector{Matrix{eltype(X)}}(undef, d)

    # map mode → index in A
    active_map = Dict(mode => idx for (idx, mode) in enumerate(active))

    for i in 1:d
        ni = size(X, i)

        if haskey(active_map, i)
            Ai = A[active_map[i]]

            if size(Ai, 2) != ni
                error("Matrix for mode $i has incompatible size $(size(Ai))")
            end

            A_full[i] = Ai
        else
            A_full[i] = Matrix(I, ni, ni)
        end
    end

    return CyclicShiftMultiplication(X, A_full)
end
"""
Strategy II : permuting

"""
function multiply_with_permutation(X::AbstractArray, A::Vector{<:AbstractMatrix}, active::Vector{Int})
    d = ndims(X)
    gap_positions = setdiff(collect(1:d), active)

    order = OptimalOrdering(X, A)
    active_ordered = active[order]
    A_ordered = A[order]

    P = vcat(active_ordered, gap_positions)

    if P == collect(1:d)
        return CyclicShiftMultiplication(X, A_ordered)
    end
    Pinv = invperm(P)
    X_perm = permutedims(X, P)
    Y = CyclicShiftMultiplication(X_perm, A_ordered)
    return permutedims(Y, Pinv)
end
