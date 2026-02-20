using BenchmarkTools
using Random
using Plots

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
        t = @benchmark NaiveMultiplication($X, $A, $P) samples = 10 evals = 1
        push!(times_ms, median(t.times) / 1e6)
    end
    return times_ms
end

# -----------------------------
# Main (4D case)
# -----------------------------
X = rand(3, 3, 3, 3, 3)

A = Vector{AbstractMatrix}(undef, 5)

A[1] = randn(40, 3)   # mode 1 (4 → 10)
A[2] = randn(240, 3)   # mode 2 (5 → 30)
A[3] = randn(360, 3)   # mode 3 (6 → 20)
A[4] = randn(120, 3)   # mode 4 (7 → 30)
A[5] = randn(1, 3)  # mode 5 (8 → 100)
println("Heuristic optimal order: ", OptimalOrdering(X, A))

perms = generate_permutations(5)
println("Number of permutations: ", length(perms))  # 24

times_ms = benchmark_permutations(perms, X, A)

# Best and worst
best_idx = argmin(times_ms)
worst_idx = argmax(times_ms)

# Print mapping (index → permutation)
println("\nPermutation index mapping:")
for i in 1:length(perms)
    println(rpad(i, 3), " → ", perms[i,])
end
println("Best permutation index = ", best_idx, "   ", perms[best_idx], "   time = ", times_ms[best_idx], " ms")
println("Worst permutation index = ", worst_idx, "   ", perms[worst_idx],
    "   time = ", times_ms[worst_idx], " ms")


# -----------------------------
# Plot (clean x-axis)
# -----------------------------

heuristic_idx = findfirst(p -> p == OptimalOrdering(X, A), perms)
colors = fill(:steelblue, length(perms))
colors[heuristic_idx] = :red   # highlight heuristic
bar(
    1:length(perms),
    times_ms,
    color=colors,
    xlabel="Permutation index",
    ylabel="Median time (ms)",
    title="NaiveMultiplication runtime over all permutations",
    legend=false,
    xticks=1:length(perms),
    size=(900, 450)
)

