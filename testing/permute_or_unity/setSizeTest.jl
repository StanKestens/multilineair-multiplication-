using LinearAlgebra
using BenchmarkTools
using Plots
using ProgressMeter
using LinearAlgebra
using BenchmarkTools
using Plots
using ProgressMeter

# ================================
# Core kernel FIRST
# ================================

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

# ================================
# Strategies
# ================================

function multiply_with_unity_fill(X, A, active)
    d = ndims(X)
    dims = size(X)

    A_full = Vector{Matrix{eltype(X)}}(undef, d)
    active_map = Dict(mode => idx for (idx, mode) in enumerate(active))

    for i in 1:d
        ni = dims[i]

        if haskey(active_map, i)
            Ai = A[active_map[i]]
            @assert size(Ai, 2) == ni
            A_full[i] = Ai
        else
            A_full[i] = Matrix(I, ni, ni)
        end
    end

    return CyclicShiftMultiplication(X, A_full)
end

function OptimalOrdering(X::AbstractArray, matrices::Vector{<:AbstractMatrix})
    ns = size(X)
    ms = [size(U, 1) for U in matrices]
    β = [1 / ns[k] - 1 / ms[k] for k in 1:length(matrices)]
    return sortperm(β)
end

function multiply_with_permutation(X, A, active)
    d = ndims(X)
    gaps = setdiff(1:d, active)

    order = OptimalOrdering(X, A)
    A = A[order]
    active = active[order]

    P = vcat(active, gaps)

    if P == collect(1:d)
        return CyclicShiftMultiplication(X, A)
    end

    Pinv = invperm(P)
    Xp = permutedims(X, P)
    Y = CyclicShiftMultiplication(Xp, A)
    return permutedims(Y, Pinv)
end

# ================================
# Tensor sizing
# ================================

function tensor_dims_from_budget(N, d)
    dims = Int[]
    remaining = N

    for k in 0:d-1
        root = d - k
        ni = floor(Int, remaining^(1 / root))
        ni = max(1, ni)
        push!(dims, ni)
        remaining = max(1, div(remaining, ni))
    end

    return Tuple(dims)
end

# ================================
# Benchmark AFTER everything exists
# ================================

function measure_times_budget(N, d, g)
    dims = tensor_dims_from_budget(N, d)
    X = randn(dims...)

    A = [randn(dims[i], dims[i]) for i in 1:(d-g)]
    active = collect(1:(d-g))

    t1 = @belapsed multiply_with_unity_fill($X, $A, $active)
    t2 = @belapsed multiply_with_permutation($X, $A, $active)

    return t1, t2, dims
end

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
orders = 3:8

results = Dict()

for d in orders
    println("\n===== d = $d =====")

    gs = 1:d-1
    t1s = zeros(length(gs), length(Ns))
    t2s = zeros(length(gs), length(Ns))

    prog = Progress(length(Ns) * length(gs), desc="d=$d", barlen=30)

    for (ni, N) in enumerate(Ns), (gi, g) in enumerate(gs)
        t1, t2, dims = measure_times_budget(N, d, g)

        t1s[gi, ni] = t1
        t2s[gi, ni] = t2

        next!(prog)
    end

    results[d] = (gs=gs, t1s=t1s, t2s=t2s)
end



Ns = [2^11, 2^12, 2^13, 2^14, 2^15, 2^16]  # sweep across cache levels

t1s = zeros(length(gs), length(Ns))
t2s = zeros(length(gs), length(Ns))

prog = Progress(length(Ns) * length(gs), desc="Benchmarking", barlen=40)

# -----------------------------------------------------------------------
# Run benchmarks
# -----------------------------------------------------------------------
for (ni, N) in enumerate(Ns), (gi, g) in enumerate(gs)
    t1, t2, dims = measure_times_budget(N, d, g)

    t1s[gi, ni] = t1
    t2s[gi, ni] = t2

    println("N=$N, dims=$dims, g=$g, t1=$(round(t1, sigdigits=3)), t2=$(round(t2, sigdigits=3))")

    next!(prog; showvalues=[(:N, N), (:g, g)])
end

plots_time = []

for d in orders
    data = results[d]
    gs = data.gs
    t1s = data.t1s
    t2s = data.t2s

    p = plot(
        xlabel="N (total elements)",
        ylabel="time (s)",
        yscale=:log10,
        title="Times (d = $d)",
        legend=:outertopright
    )

    for (gi, g) in enumerate(gs)
        plot!(p, Ns, t1s[gi, :],
            label="t₁ g=$g",
            lw=2,
            marker=:circle
        )

        plot!(p, Ns, t2s[gi, :],
            label="t₂ g=$g",
            lw=2,
            marker=:diamond,
            linestyle=:dash
        )
    end

    push!(plots_time, p)
end

plot(plots_time..., layout=(length(orders), 1), size=(900, 300 * length(orders)))










