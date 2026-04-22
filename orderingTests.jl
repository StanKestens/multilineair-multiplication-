using BenchmarkTools
using Random
using Plots
using Measures  # nodig voor mm

include("tensor.jl")
include("ordering.jl")
include("naive.jl")

# -----------------------------
# Generate all permutations
# -----------------------------
function generate_permutations(n::Int)
    if n == 1
        return [[1]]
    else
        perms = Vector{Vector{Int}}()
        for p in generate_permutations(n - 1)
            for i in 1:n
                new_perm = copy(p)
                insert!(new_perm, i, n)
                push!(perms, new_perm)
            end
        end
        return perms
    end
end

# -----------------------------
# Benchmark all permutations
# -----------------------------
function benchmark_permutations(perms, X, A)
    times_ms = Float64[]
    for P in perms
        t = @benchmark NaiveMultiplication($X, $A, $P) samples = 5 evals = 1
        push!(times_ms, median(t.times) / 1e6)
    end
    return times_ms
end

# -----------------------------
# Count operations
# -----------------------------
function count_operations(X, A, P)
    dims = collect(size(X))
    total_ops = 0

    for mode in P
        r = size(A[mode], 1)
        n = dims[mode]
        other = prod(dims) ÷ n

        total_ops += r * n * other
        dims[mode] = r
    end

    return total_ops
end

# -----------------------------
# Main (4D case)
# -----------------------------
X = rand(50,50,50,50)

A = Vector{AbstractMatrix}(undef, 4)
A[1] = randn(200, 50)
A[2] = randn(150, 50)
A[3] = randn(100, 50)
A[4] = randn(1,  50)

heuristic = OptimalOrdering(X, A)
println("Heuristic optimal order: ", heuristic)

perms = generate_permutations(4)
println("Number of permutations: ", length(perms))

times_ms = benchmark_permutations(perms, X, A)
ops = [count_operations(X, A, p) for p in perms]

best_idx = argmin(times_ms)
worst_idx = argmax(times_ms)

println("\nPermutation index mapping:")
for i in 1:length(perms)
    println(rpad(i, 3), " → ", perms[i])
end

println("\nBest permutation index = ", best_idx,
    "   ", perms[best_idx],
    "   time = ", times_ms[best_idx], " ms")

println("Worst permutation index = ", worst_idx,
    "   ", perms[worst_idx],
    "   time = ", times_ms[worst_idx], " ms")
# -----------------------------
# Plot (clean legend + right axis)
# -----------------------------

heuristic_idx = findfirst(p -> p == heuristic, perms)

perm_labels = [join(p, "\n") for p in perms]
x = 1:length(perms)

# Scale operations visually
scale_factor = maximum(times_ms) / maximum(ops)
ops_scaled = ops .* scale_factor

bar_width = 0.35

# --- Build color vectors ---
time_colors = fill(:steelblue, length(perms))
ops_colors = fill(:seagreen, length(perms))

time_colors[heuristic_idx] = Symbol("#804080")
ops_colors[heuristic_idx] = Symbol("#A06F0D")

# --- Plot runtime bars (left axis) ---
p = bar(
    x .- bar_width / 2,
    times_ms,
    bar_width=bar_width,
    color=time_colors,
    label="",   # we handle legend manually
    xlabel="Permutation",
    ylabel="Median time (ms)",
    xticks=(x, perm_labels),
    legend=:topleft,
    size=(1200, 500),
    left_margin=8mm,
    right_margin=8mm,
    bottom_margin=18mm
)

# --- Plot scaled FLOP bars ---
bar!(
    x .+ bar_width / 2,
    ops_scaled,
    bar_width=bar_width,
    color=ops_colors,
    label=""
)

# --- Right axis with TRUE FLOP scale ---
p2 = twinx()

plot!(
    p2,
    x,
    ops,
    alpha=0,
    ylabel="Operations (FLOPs)",
    label=""
)

# -----------------------------
# Manual clean legend
# -----------------------------

# Blue legend entry
scatter!([NaN], [NaN],
    markercolor=:steelblue,
    markersize=8,
    label="Time (ms) — left axis"
)

# Green legend entry
scatter!([NaN], [NaN],
    markercolor=:seagreen,
    markersize=8,
    label="FLOPs — right axis"
)

# Red legend entry
scatter!([NaN], [NaN],
    markercolor=Symbol("#804080"),
    markersize=8,
    label="Heuristic time"
)
scatter!([NaN], [NaN],
    markercolor=Symbol("#A06F0D"),
    markersize=8,
    label="Hueristic operations"
)
display(p)