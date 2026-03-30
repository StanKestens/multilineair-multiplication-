using LinearAlgebra
using Plots
using ProgressMeter

include("cyclicShift.jl")

function measure_speedup(n, d, g; reps=3)
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

    return t1 / t2
end

d = 5
ns = [22, 26, 30, 34, 36]
gs = 1:d-1

speedups = zeros(length(gs), length(ns))
prog = Progress(length(ns) * length(gs), desc="Benchmarking... ", barlen=40)

for (ni, n) in enumerate(ns), (gi, g) in enumerate(gs)
    speedups[gi, ni] = measure_speedup(n, d, g)
    next!(prog; showvalues=[(:n, n), (:g, g), (:speedup, round(speedups[gi, ni], digits=2))])
end

# --- color: diverging around 1 ---
# log-transform so 0.5 and 2.0 are equidistant from 1
log_speedups = log.(speedups)
lim = max(abs(minimum(log_speedups)), abs(maximum(log_speedups)))

p = heatmap(ns, collect(gs), log_speedups,
    xlabel="n  (tensor dim)",
    ylabel="g  (# gaps)",
    title="Speedup t₁/t₂  —  green = permute wins, red = identity wins",
    color=:RdYlGn,
    clims=(-lim, lim),       # symmetric around 0 (= speedup of 1)
    colorbar_title="log speedup",
    xticks=ns,
    yticks=collect(gs),
    size=(800, 450))

# break-even line: gn = 2
n_cont = range(ns[1], ns[end], length=300)
plot!(p, n_cont, 2 ./ n_cont,
    lw=2, lc=:black, ls=:dash,
    label="theory: gn = 2",
    ylims=(0.5, length(gs) + 0.5))

# annotate raw speedup (not log)
for (ni, n) in enumerate(ns), (gi, g) in enumerate(gs)
    annotate!(p, n, g, text(string(round(speedups[gi, ni], digits=1)), 7, :black))
end

display(p)
savefig(p, "speedup_heatmap.png")