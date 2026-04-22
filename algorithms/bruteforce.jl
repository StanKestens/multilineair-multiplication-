using TensorToolbox
using LinearAlgebra

function bruteforce(X::AbstractArray, A::Vector{<:AbstractMatrix})
    d = ndims(X)

    K = A[d]
    for k in (d-1):-1:1
        K = kron(K, A[k])
    end

    y = K *vec(X)

    outsz = ntuple(i -> size(A[i], 1), d)
    return reshape(y, outsz)
end