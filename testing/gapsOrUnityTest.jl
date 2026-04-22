using BenchmarkTools
using Random
using LinearAlgebra
using Plots
using Combinatorics
using ProgressMeter

include("tensor.jl")
include("ordering.jl")
include("naive.jl")
include("cyclicShift.jl")

function benchmark_cyclic(d, g; trials=1)
    min_grootte_tensor = 2
    max_grootte_tensor = 3
    min_grootte_matrix = 2
    max_grootte_matrix = 3
    # Random tensor
    dimensies = rand(min_grootte_tensor:max_grootte_tensor, d)
    X = randn(dimensies...)

    A = [randn(rand(min_grootte_matrix:max_grootte_matrix), dimensies[i]) for i in 1:d]
    perm_identity = collect(1:d)

    t1_total = 0.0
    t2_total = 0.0
    count = 0
    all_combos = collect(combinations(1:d, g))
    random_combos = rand(all_combos, min(5, length(all_combos)))
    for gap_positions in random_combos
        println(gap_positions)
        # OPTION 1
        A_unity = copy(A)
        for i in gap_positions
            n_i = dimensies[i]
            A_unity[i] = Matrix(I, n_i, n_i)
        end

        t1 = @belapsed CyclicShiftMultiplication($X, $A_unity, $perm_identity)

        # OPTION 2: Permute gaps to end
        println("optie 2")
        OP = OptimalOrdering(X, A)
        P = vcat(setdiff(OP, gap_positions), gap_positions)
        X_perm = permutedims(X, P)
        A_perm = A[P]
        A_final = A_perm[1:end-length(gap_positions)]
        println(length(A_final))
        t2 = @belapsed begin
            Y = CyclicShiftMultiplication($X_perm, $A_final, $perm_identity)
            permutedims(Y, invperm($P))
        end

        t1_total += t1
        t2_total += t2
        count += 1
    end
    # Return average over all gap placements
    return t1_total / count, t2_total / count
end

ds = 9:15
g = 5 # try 1 gap first

function run_benchmarks(ds, g)
    times_unity = Float64[]
    times_perm = Float64[]

    total = length(ds)
    prog = Progress(total, desc="Benchmarking...")

    for d in ds
        println("Running dimension d = $d ")
        t1, t2 = benchmark_cyclic(d, g)

        push!(times_unity, t1)
        push!(times_perm, t2)

        next!(prog)   #  update progress bar
    end

    return times_unity, times_perm
end
times_unity, times_perm = run_benchmarks(ds, g)

display(plot(ds, times_unity,
    label="Unity Matrix",
    xlabel="Tensor Order d",
    ylabel="Time (seconds)",
    lw=2))

plot!(ds, times_perm,
    label="Permute Strategy",
    lw=2,
    yscale=:log10)

display(current())
