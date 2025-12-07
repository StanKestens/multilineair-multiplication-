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
        t = @benchmark NaiveMultiplication($X, $A, $P) samples=50 evals=1
        push!(times_ms, median(t.times) / 1e6)
    end
    return times_ms
end

# -----------------------------
# Main (4D case)
# -----------------------------
X = randn(50,50,50,50) 

A = MatrixCell([
    randn(1,50),
    randn(5,50),
    randn(500,50),
    randn(1500,50)
])

println("Heuristic optimal order: ", OptimalOrdering(X, A))

perms = generate_permutations(4)
println("Number of permutations: ", length(perms))  # 24

times_ms = benchmark_permutations(perms, X, A)

# Best and worst
best_idx  = argmin(times_ms)
worst_idx = argmax(times_ms)

# Print mapping (index → permutation)
println("\nPermutation index mapping:")
for i in 1:length(perms)
    println(rpad(i, 3), " → ", perms[i])
end
println("Best permutation index = ", best_idx,  "   ", perms[best_idx],
        "   time = ", times_ms[best_idx], " ms")
println("Worst permutation index = ", worst_idx, "   ", perms[worst_idx],
        "   time = ", times_ms[worst_idx], " ms")


# -----------------------------
# Plot (clean x-axis)
# -----------------------------
bar(
    1:length(perms),
    times_ms,
    xlabel = "Permutation index (1–24)",
    ylabel = "Median time (ms)",
    title  = "NaiveMultiplication runtime over all 4! permutations",
    legend = false,
    xticks = 1:24,   # clean numbered x-axis
    xrotation = 0,
    size = (900, 450)
)
