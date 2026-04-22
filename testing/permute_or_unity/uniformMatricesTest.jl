using LinearAlgebra
using Plots
using ProgressMeter
using BenchmarkTools
include("../algorithms/cyclicShift.jl")
function max_n_for_d(d, max_elems)
    return floor(Int, max_elems^(1 / d))
end
# -----------------------------------------------------------------------
# Benchmark: returns (α, γ, speedup, t_full)
#   α  = permutation cost  (permutedims forward + backward)
#   γ  = reduced matmul cost  (CyclicShiftMultiplication on active dims only)
#   t_full = full matmul with identity matrices in gap positions  (= t₁)
#   speedup = t_full / (α + γ)
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Modify measure function to return raw times
# -----------------------------------------------------------------------
function measure_times(n, d, g; reps=3)
    X = randn(ntuple(_ -> n, d)...)
    A = [randn(n, n) for _ in 1:d]
    perm_id = collect(1:d)
    active = collect(1:d-g)
    gaps = collect(d-g+1:d)

    A_unity = copy(A)
    for i in gaps
        A_unity[i] = Matrix(I, n, n)
    end

    t1 = minimum(@elapsed(CyclicShiftMultiplication(X, A_unity, perm_id)) for _ in 1:reps)

    t2 = minimum(@elapsed(begin
        P = vcat(active, gaps)
        Pinv = invperm(P)
        X_perm = permutedims(X, P)
        A_act = A[active]
        Y = CyclicShiftMultiplication(X_perm, A_act, perm_id)
        permutedims(Y, Pinv)
    end) for _ in 1:reps)

    return t1, t2
end

# -----------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------
d = 3
ns = [5, 8, 10, 15, 20, 40, 60, 80, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400, 440]
gs = 1:d-1

t1s = zeros(length(gs), length(ns))
t2s = zeros(length(gs), length(ns))

prog = Progress(length(ns) * length(gs), desc="Benchmarking... ", barlen=40)

for (ni, n) in enumerate(ns), (gi, g) in enumerate(gs)
    t1s[gi, ni], t2s[gi, ni] = measure_times(n, d, g)
    next!(prog; showvalues=[(:n, n), (:g, g)])
end

# -----------------------------------------------------------------------
# Plot: one panel per g
# -----------------------------------------------------------------------
ng = length(gs)
pal = palette(:Dark2, ng)

plots = map(enumerate(gs)) do (gi, g)
    p = plot(
        ns, t1s[gi, :],
        label="t₁ (identity fill)",
        lw=2,
        marker=:circle,
        color=:red,
        yscale=:log10,
        xlabel="n",
        ylabel="time (s)",
        title="g = $g",
    )

    plot!(p,
        ns, t2s[gi, :],
        label="t₂ (permute + reduced)",
        lw=2,
        marker=:diamond,
        color=:green,
    )

    p
end
# -----------------------------------------------------------------------
# Plot: speedup = t₁ / t₂
# -----------------------------------------------------------------------
speedups = t1s ./ t2s

plots_speed = map(enumerate(gs)) do (gi, g)
    p = plot(
        ns, speedups[gi, :],
        label="speedup",
        lw=2,
        marker=:circle,
        color=:blue,
        xlabel="n",
        ylabel="t₁ / t₂",
        title="g = $g",
    )

    # break-even line
    hline!(p, [1.0],
        lw=1, lc=:black, ls=:dash,
        label="break-even")

    p
end

p_speed = plot(
    legend=:outertopright,
    plots_speed...,
    layout=(2, ceil(Int, ng / 2)),
    size=(1200, 700),
    margin=10Plots.mm
)

display(p_speed)
# layout
p_final = plot(
    legend=:outertopright,
    plots...,
    layout=(2, ceil(Int, ng / 2)),   # 2 rows instead of 1 long row
    size=(1200, 700),              # much bigger canvas
    margin=10Plots.mm              # more breathing room
)

display(p_final)

