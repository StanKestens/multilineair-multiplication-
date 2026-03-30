using LinearAlgebra
using Plots
using ProgressMeter
using Statistics

function measure_ratio(n, r; reps=10)
    # Tensor
    X = randn(n, n, n, n, n)

    # Simulated reshape: (n^2 × n)
    K = n^4

    A = randn(n, r)


    # -----------------------------
    # Permutation cost
    # -----------------------------
    t_perm = minimum(@elapsed begin
        permutedims(X, (2, 1, 5, 4, 3))
    end for _ in 1:reps)

    # -----------------------------
    # Realistic matmul
    # -----------------------------
    Xmat = randn(K, n)

    t_mm = minimum(@elapsed begin
        Xmat * A
    end for _ in 1:reps)

    return t_perm / t_mm
end

# -----------------------------
# Sweep
# -----------------------------
ns = [10, 20, 30, 40]
r = 8   # small rank (typical use case)

ratios = zeros(length(ns))

prog = Progress(length(ns), desc="Benchmarking")

for (i, n) in enumerate(ns)
    ratios[i] = measure_ratio(n, r)
    next!(prog, showvalues=[(:n, n), (:ratio, round(ratios[i], digits=3))])
end

println("\nAverage ratio = ", mean(ratios))

# -----------------------------
# Plot
# -----------------------------
plot(ns, ratios,
    marker=:o,
    lw=2,
    xlabel="n",
    ylabel="time(permute) / time(realistic matmul)",
    title="Permutation vs Realistic Tensor Matmul",
    label="ratio"
)

hline!([1.0], linestyle=:dash, label="break-even")

display(current())