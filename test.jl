using BenchmarkTools
using Random
using LinearAlgebra
using Statistics
using Printf
using DataFrames
using CSV

# ------------------------------------------------------------
# 0. Helpers: unfold/fold (vervang door jullie eigen versies)
# ------------------------------------------------------------

"""
unfold(X, mode): matricize tensor X along `mode` (1-based),
result has size (size(X,mode), prod(other dims)).
Uses permutedims + reshape.
"""
function unfold(X::AbstractArray, mode::Int)
    N = ndims(X)
    @assert 1 ≤ mode ≤ N
    perm = (mode, setdiff(1:N, mode)...)
    Y = permutedims(X, perm)
    return reshape(Y, size(X, mode), :)
end

"""
fold(M, mode, dims): inverse of unfold, where M is (dims[mode], prod(other dims)).
Returns tensor of shape `dims`.
"""
function fold(M::AbstractMatrix, mode::Int, dims::Vector{Int})
    N = length(dims)
    @assert 1 ≤ mode ≤ N
    other = setdiff(1:N, mode)
    Y = reshape(M, dims[mode], dims[other]...)
    invp = invperm((mode, other...))
    return permutedims(Y, invp)
end

# ------------------------------------------------------------
# 1. Operaties om te benchmarken
# ------------------------------------------------------------

# permdims: algemene permutatie (worst-case data beweging)
perm_ops(X, perm) = permutedims(X, perm)

# transpose/adjoint: alleen zinvol voor 2D; voor tensor kan je bv. permute 1<->2 doen
transpose_2d(A) = transpose(A)     # lazy wrapper (geen copy) in veel gevallen
adjoint_2d(A)   = adjoint(A)       # idem, voor complex conj

# copy: dwingt materialisatie
copy_op(X) = copy(X)

# ------------------------------------------------------------
# 2. Benchmark runner: tijd + allocaties
# ------------------------------------------------------------

"""
bench_one(opname, f, args...; samples=, seconds=)
Return tuple: (time_ns, alloc_bytes, allocs)
"""
function bench_one(opname::String, f, args...; samples=200, seconds=1.0)
    # BenchmarkTools: meet robuust + alloc info
    b = @benchmark $f($(args...)) samples=$samples seconds=$seconds
    t = median(b).time                 # ns
    bytes = median(b).memory           # bytes
    allocs = median(b).allocs
    return (opname, t, bytes, allocs)
end

# ------------------------------------------------------------
# 3. Testcases (vormen + permutaties)
# ------------------------------------------------------------

# Kies vormen die zowel "vierkant" als "scheef" zijn (memory layout impact)
function default_shapes()
    return [
        (20, 20, 20),
        (40, 20, 10),
        (30, 30, 10, 5),
        (16, 16, 16, 16),
        (64, 16, 8, 4),
    ]
end

# Enkele representatieve permutaties (identity, swap, reverse, random)
function default_perms(N::Int)
    perms = Vector{Vector{Int}}()
    push!(perms, collect(1:N))                 # identity
    if N ≥ 2
        p = collect(1:N); p[1], p[2] = p[2], p[1]
        push!(perms, p)                        # swap 1<->2
    end
    push!(perms, collect(N:-1:1))              # reverse
    rng = MersenneTwister(1)
    push!(perms, randperm(rng, N))             # random
    return perms
end

# ------------------------------------------------------------
# 4. Hoofdbenchmark: tijd + allocaties + output tabel
# ------------------------------------------------------------

function run_benchmarks(; shapes=default_shapes(), T=Float64, seed=1,
                        samples=200, seconds=1.0, out_csv="tensor_ops_bench.csv")

    rng = MersenneTwister(seed)
    rows = DataFrame(
        op = String[],
        dims = String[],
        nd = Int[],
        mode = Union{Missing,Int}[],
        perm = Union{Missing,String}[],
        time_ns = Int[],
        alloc_bytes = Int[],
        allocs = Int[]
    )

    for shp in shapes
        X = randn(rng, T, shp...)
        dims_vec = collect(shp)
        N = length(dims_vec)

        dims_str = join(dims_vec, "×")

        # ---- permdims benchmarks (meerdere permutaties) ----
        for p in default_perms(N)
            opname = "permutedims"
            (op, t, bytes, allocs) = bench_one(opname, perm_ops, X, p; samples=samples, seconds=seconds)
            push!(rows, (op, dims_str, N, missing, join(p, ","), t, bytes, allocs))
        end

        # ---- unfold/fold per mode ----
        for mode in 1:N
            (op, t, bytes, allocs) = bench_one("unfold", unfold, X, mode; samples=samples, seconds=seconds)
            push!(rows, (op, dims_str, N, mode, missing, t, bytes, allocs))

            # fold: maak eerst een matrix met correcte size
            M = unfold(X, mode)  # éénmalig buiten benchmark (niet meetellen)
            (op2, t2, bytes2, allocs2) = bench_one("fold", fold, M, mode, dims_vec; samples=samples, seconds=seconds)
            push!(rows, (op2, dims_str, N, mode, missing, t2, bytes2, allocs2))
        end

        # ---- copy (materialisatie) ----
        (op, t, bytes, allocs) = bench_one("copy", copy_op, X; samples=samples, seconds=seconds)
        push!(rows, (op, dims_str, N, missing, missing, t, bytes, allocs))

        # ---- transpose/adjoint (alleen voor 2D tensors) ----
        if N == 2
            A = X
            (op, t, bytes, allocs) = bench_one("transpose", transpose_2d, A; samples=samples, seconds=seconds)
            push!(rows, (op, dims_str, N, missing, missing, t, bytes, allocs))

            (op, t, bytes, allocs) = bench_one("adjoint", adjoint_2d, A; samples=samples, seconds=seconds)
            push!(rows, (op, dims_str, N, missing, missing, t, bytes, allocs))

            # forceer materialisatie van transpose/adjoint
            (op, t, bytes, allocs) = bench_one("copy(transpose)", x -> copy(transpose(x)), A; samples=samples, seconds=seconds)
            push!(rows, (op, dims_str, N, missing, missing, t, bytes, allocs))
        end
    end

    CSV.write(out_csv, rows)
    return rows
end

# ------------------------------------------------------------
# 5. Run (als script)
# ------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    df = run_benchmarks(samples=150, seconds=0.5, out_csv="tensor_ops_bench.csv")
    println(df)
    @printf("\nWrote results to tensor_ops_bench.csv\n")
end