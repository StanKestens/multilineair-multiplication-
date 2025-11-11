using LinearAlgebra
using TensorToolbox
"""
Function to decide to optimale order of ttm, for theory, see Proposition 3.25 (Multi-TTM Ordering, Fackler, 2019) 
Input : X::AbstractArray, Our representation of a tensor
        matrices::MatrixCell , a list of matrices, represented using MatrixCell, which Lana Perisa also used
Output: order , the results by the given formula

"""
function OptimalOrdering(X::AbstractArray, matrices::MatrixCell)
    ns = size(X)
    ms = [size(U, 1) for U in matrices]
    β = [1/ns[k] - 1/ms[k] for k in 1:length(matrices)]
    order = sortperm(β)  
    return order
end


