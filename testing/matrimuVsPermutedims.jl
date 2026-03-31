using LinearAlgebra
using ProgressMeter
using Plots
using Statistics

function measure_alpha_gamma_large(n; reps=5)
    X = randn(n, n)
    A = randn(n, n)

    # warm up
    permutedims(X, (2, 1))
    A * X

    alpha = minimum(@elapsed(permutedims(X, (2, 1))) for _ in 1:reps)
    gamma = minimum(@elapsed(A * X) for _ in 1:reps)

    return alpha, gamma, alpha / gamma
end

ns_large = [100, 300, 500, 1000, 2000, 3000, 5000, 7000, 10000]

results_large = zeros(length(ns_large))
alphas_large = zeros(length(ns_large))
gammas_large = zeros(length(ns_large))

prog = Progress(length(ns_large), desc="Large-n benchmark... ", barlen=40, showspeed=true)

for (ni, n) in enumerate(ns_large)
    a, g, r = measure_alpha_gamma_large(n)
    alphas_large[ni] = a
    gammas_large[ni] = g
    results_large[ni] = r

    next!(prog; showvalues=[
        (:n, n),
        (:α, round(a, sigdigits=3)),
        (:γ, round(g, sigdigits=3)),
        (:α_γ, round(r, digits=4)),
        (:n_times_ratio, round(n * r, digits=2))
    ])
end

# ── compute C properly ────────────────────────────────────────────────────────
C_vals = ns_large .* results_large

rel_change = abs.(diff(C_vals)) ./ C_vals[2:end]
tol = 0.03

stable_idx = findfirst(i -> all(rel_change[i:end] .< tol),
    1:length(rel_change))

if stable_idx === nothing
    println("No clear convergence yet")
    C_est = mean(C_vals[end-2:end])   # fallback
else
    C_est = mean(C_vals[stable_idx+1:end])
end

println("Estimated C = ", round(C_est, digits=3))

# ── plot α/γ ──────────────────────────────────────────────────────────────────
n_fine = range(ns_large[1], ns_large[end], length=500)

p1 = plot(
    title="α/γ vs n  (d=2, large n)",
    xlabel="n",
    ylabel="α/γ",
    xscale=:log10,
    yscale=:log10,
    legend=:topright,
    size=(700, 450)
)

scatter!(p1, ns_large, results_large,
    label="measured",
    marker=:circle,
    markersize=6,
    color=:royalblue)

plot!(p1, ns_large, results_large,
    label=false, lw=2, color=:royalblue)

plot!(p1, n_fine, 1 ./ n_fine,
    label="1/n",
    lw=2, ls=:dash, color=:black)

plot!(p1, n_fine, C_est ./ n_fine,
    label="C/n  (C = $(round(C_est, digits=1)))",
    lw=2, ls=:dot, color=:red)

# ── stability plot ────────────────────────────────────────────────────────────
p2 = plot(
    title="n · (α/γ) convergence",
    xlabel="n",
    ylabel="C = n · α/γ",
    xscale=:log10,
    legend=:bottomright,
    size=(700, 450)
)

scatter!(p2, ns_large, C_vals,
    marker=:circle,
    markersize=6,
    color=:royalblue,
    label="n · α/γ")

plot!(p2, ns_large, C_vals,
    lw=2, color=:royalblue, label=false)

hline!(p2, [C_est],
    lw=2, ls=:dash, color=:red,
    label="C = $(round(C_est, digits=1))")

# ── combine ───────────────────────────────────────────────────────────────────
final = plot(p1, p2, layout=(1, 2), size=(1300, 480))

display(final)
savefig(final, "alpha_gamma_large_n.png")