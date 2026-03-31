using LinearAlgebra
using Plots
using ProgressMeter
using Random
using Printf

include("cyclicShift.jl")
include("ordering.jl") # Assuming OptimalOrdering is here

# -----------------------------------------------------------------------
# Helper: Find dimension size `n` for a target memory footprint
# -----------------------------------------------------------------------
function get_n_for_target_mb(d::Int, target_mb::Real)
    target_bytes = target_mb * 1024^2
    target_elements = target_bytes / sizeof(Float64)
    n = floor(Int, target_elements^(1 / d))
    return n
end

# -----------------------------------------------------------------------
# Benchmarking Core
# -----------------------------------------------------------------------
function measure_times(n, d, g; trials=3)
    dimensies = ntuple(_ -> n, d)
    X = randn(dimensies...)

    # IMPORTANT: Using n x n ensures the tensor stays exactly ~100 MB 
    # throughout the entire contraction. If you use 2*n, memory will blow up 
    # exponentially for higher dimensions (e.g., d=8 -> 2^8 times larger).
    A = [randn(2 * n, n) for _ in 1:d]
    perm_identity = collect(1:d)

    t1_acc = 0.0
    t2_acc = 0.0

    for _ in 1:trials
        all_indices = collect(1:d)
        gap_positions = Random.shuffle(all_indices)[1:g]

        # --- STRATEGY 1: Unity Matrices (C1) ---
        A_unity = copy(A)
        for i in gap_positions
            A_unity[i] = Matrix{Float64}(I, n, n)
        end

        t1 = @elapsed CyclicShiftMultiplication(X, A_unity, perm_identity)

        # --- STRATEGY 2: Permutation + Reduced MatMul (C2) ---
        t2 = @elapsed begin
            OP = OptimalOrdering(X, A)
            P = vcat(setdiff(OP, gap_positions), gap_positions)
            Pinv = invperm(P)

            X_p = permutedims(X, P)
            # Grab matrices belonging to the active dimensions
            A_active = A[P[1:end-length(gap_positions)]]

            Y = CyclicShiftMultiplication(X_p, A_active, perm_identity)
            permutedims(Y, Pinv)
        end

        t1_acc += t1
        t2_acc += t2
    end

    return t1_acc / trials, t2_acc / trials
end

# -----------------------------------------------------------------------
# Optimized Sweep for High Stability
# -----------------------------------------------------------------------
TARGET_MB = 400.0  # Increased to 500 MB for better arithmetic/overhead ratio
ds = 4:9         # Let's go up to order 10
TRIALS = 10        # More trials to smooth out the "shuffled" noise

results_t1 = Dict{Int,Vector{Float64}}()
results_t2 = Dict{Int,Vector{Float64}}()
results_ns = Dict{Int,Int}()

# Pre-calculate total iterations
total_iters = sum(d - 1 for d in ds)
prog = Progress(total_iters, desc="Gathering Data... ", barlen=40)

# Warmup
measure_times(4, 3, 1; trials=1)

for d in ds
    n = get_n_for_target_mb(d, TARGET_MB)
    results_ns[d] = n

    # We collect all trials to take the median or a better average
    t1_arr = Float64[]
    t2_arr = Float64[]

    for g in 1:d-1
        # Run measure_times with more trials for a cleaner average
        t1, t2 = measure_times(n, d, g; trials=TRIALS)
        push!(t1_arr, t1)
        push!(t2_arr, t2)

        next!(prog; showvalues=[(:d, d), (:n, n), (:g, g)])
    end

    results_t1[d] = t1_arr
    results_t2[d] = t2_arr
end
# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------
plots_raw = []
plots_speed = []

for d in ds
    n = results_ns[d]
    gs = 1:d-1
    t1s = results_t1[d]
    t2s = results_t2[d]
    speedups = t1s ./ t2s

    tensor_mb = round((n^d) * 8 / 1024^2, digits=1)

    # --- Raw Times Plot ---
    p_raw = plot(
        gs, t1s,
        label="t₁ (identity fill)",
        lw=2, marker=:circle, color=:red,
        xlabel="Number of Gaps (g)",
        ylabel="Time (s)",
        title="d = $d, n = $n (~$tensor_mb MB)",
        legend=:topright
    )
    plot!(p_raw, gs, t2s,
        label="t₂ (permute + reduced)",
        lw=2, marker=:diamond, color=:green
    )
    push!(plots_raw, p_raw)

    # --- Speedup Plot ---
    p_speed = plot(
        gs, speedups,
        label="speedup (t₁/t₂)",
        lw=2, marker=:circle, color=:blue,
        xlabel="Number of Gaps (g)",
        ylabel="Speedup factor",
        title="d = $d, n = $n (~$tensor_mb MB)",
        legend=:topright
    )
    hline!(p_speed, [1.0], lw=1, lc=:black, ls=:dash, label="break-even")
    push!(plots_speed, p_speed)
end

# Layout constraints based on number of ds we tested
num_plots = length(ds)
cols = 2
rows = ceil(Int, num_plots / cols)

p_final_raw = plot(plots_raw..., layout=(rows, cols), size=(1000, 400 * rows), margin=5Plots.mm)
p_final_speed = plot(plots_speed..., layout=(rows, cols), size=(1000, 400 * rows), margin=5Plots.mm)

println("Displaying Speedup Plots...")
display(p_final_speed)

println("Displaying Raw Time Plots...")
display(p_final_raw)