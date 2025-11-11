using LinearAlgebra
using TensorToolbox
"""
Function to decide to optimale order of ttm, for theory, see Proposition 3.25 (Multi-TTM Ordering, Fackler, 2019) 
Input : X::AbstractArray, Our representation of a tensor
        matrices::MatrixCell , a list of matrices, represented using MatrixCell, which Lana Perisa also used
Output: Y, β ; these are temporary just to test the implemtation
        order , the result given by the formula

"""
function OptimalOrdering(X::AbstractArray, matrices::MatrixCell)
    ns = size(X)
    ms = [size(U, 1) for U in matrices]
    β = [1/ns[k] - 1/ms[k] for k in 1:length(matrices)]
    order = sortperm(β)  

    Y = X
    for k in order
        Y = ttm(Y, matrices[k], k)
    end

    return Y, order, β
end
X = rand(3,4,5)
matrices = MatrixCell([
    rand(5,3),   # for mode 1
    rand(4,4),   # for mode 2
    rand(3,5)    # for mode 3
])
Y, order, β = multi_ttm_optimal(X, matrices)

println("Optimal order: ", order)
println("β values: ", β)
println("Resulting tensor size: ", size(Y))
