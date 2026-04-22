using LinearAlgebra
using Plots
using ProgressMeter
using BenchmarkTools
using TensorToolbox
using Strided
include("../../algorithms/cyclicShift.jl")   # adjust path as needed

# -----------------------------------------------------------------------
# Benchmark: compares the two strategies for a given gap count g
#   t1 = multiply_with_unity_fill   (Strategy I)
#   t2 = multiply_with_permutation  (Strategy II)
# -----------------------------------------------------------------------
#why 
function measure_times(n, d, g, reps=3)
    p = 2 # test with 2, 0.2
    X = randn(ntuple(_ -> n, d)...)
    A = [randn(floor(Int, p * n), n) for _ in 1:d]
    active = collect(1:d-g)

    t1 = @belapsed multiply_with_unity_fill($X, $A[$active], $active)
    t2 = @belapsed multiply_with_permutation($X, $A[$active], $active)

    return t1, t2
end

# -----------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------
#run : d= 6, ns = [3... 20]
#d=7 , ns = [3, 5, 8, 10, 12]
#maybe d=8, ns = [3, 5, 6, 8, 9]
d = 5
ns = [3, 7, 10, 16, 20]
gs = 1:d-1

t1s = zeros(length(gs), length(ns))
t2s = zeros(length(gs), length(ns))

prog = Progress(length(ns) * length(gs), desc="Benchmarking... ", barlen=40)

for (ni, n) in enumerate(ns), (gi, g) in enumerate(gs)
    t1s[gi, ni], t2s[gi, ni] = measure_times(n, d, g)
    next!(prog; showvalues=[(:n, n), (:g, g)])
end

# -----------------------------------------------------------------------
# Plot: raw times, one panel per g
# -----------------------------------------------------------------------
ng = length(gs)

plots_time = map(enumerate(gs)) do (gi, g)
    p = plot(
        ns, t1s[gi, :],
        label="t₁ (unity fill)",
        lw=2, marker=:circle, color=:red,
        yscale=:log10,
        xlabel="n", ylabel="time (s)",
        title="g = $g",
    )
    plot!(p,
        ns, t2s[gi, :],
        label="t₂ (permute + reduced)",
        lw=2, marker=:diamond, color=:green,
    )
    p
end

p_time = plot(
    plots_time...,
    layout=(2, ceil(Int, ng / 2)),
    legend=:outertopright,
    size=(1200, 700),
    margin=10Plots.mm,
)
display(p_time)

# -----------------------------------------------------------------------
# Plot: speedup = t₁ / t₂, one panel per g
# -----------------------------------------------------------------------
speedups = t1s ./ t2s

plots_speed = map(enumerate(gs)) do (gi, g)
    p = plot(
        ns, speedups[gi, :],
        label="speedup",
        lw=2, marker=:circle, color=:blue,
        xlabel="n", ylabel="t₁ / t₂",
        title="g = $g",
    )
    hline!(p, [1.0], lw=1, lc=:black, ls=:dash, label="break-even")
    p
end

p_speed = plot(
    plots_speed...,
    layout=(2, ceil(Int, ng / 2)),
    legend=:outertopright,
    size=(1200, 700),
    margin=10Plots.mm,
)
display(p_speed)