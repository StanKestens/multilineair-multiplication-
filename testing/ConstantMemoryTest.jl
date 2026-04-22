"""
Verdeel de priemfactoren van N zo gelijk mogelijk over d dimensies,
zodat het product exact N blijft.
"""
function factor_based_shape(N::Int, d::Int)
    dims = ones(Int, d)
    n = N
    factors = Int[]

    p = 2
    while p * p <= n
        while n % p == 0
            push!(factors, p)
            n ÷= p
        end
        p += 1
    end
    if n > 1
        push!(factors, n)
    end

    # verdeel factoren steeds naar de kleinste huidige dimensie
    for f in sort(factors, rev=true)
        i = argmin(dims)
        dims[i] *= f
    end

    return Tuple(sort(dims, rev=true))
end
