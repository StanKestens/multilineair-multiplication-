using TensorToolbox
using Random
using Plots
using Statistics
using Printf
using LinearAlgebra

include("tensor.jl")
include("ordering.jl")
include("constantMemoryTest.jl")

# ============================================================
# 1. Data structs
# ============================================================

struct SplitTimes
    planning_ms::Float64
    compute_ms::Float64
    reorg_ms::Float64
end

struct BenchmarkResults
    dims::Vector{Int}
    methods::Vector{Symbol}
    stats::Dict{Symbol,Vector{SplitTimes}}
end

# ============================================================
# 2. Helper
# ============================================================

OptimalOrdering(X, A) = sortperm([1 / size(X, k) - 1 / size(A[k], 1) for k in eachindex(A)])

# ============================================================
# 3. Timed methods
# ============================================================

function NaiveMultiplication_timed(X::AbstractArray, A::Vector{<:AbstractMatrix}, order::Vector{Int})
    t_plan = 0.0
    t_comp = 0.0
    t_reorg = 0.0

    for mode in order
        t0 = time_ns()
        Xmat = unfold(X, mode)
        t1 = time_ns()
        Xmat = A[mode] * Xmat
        t2 = time_ns()
        sz = collect(size(X))
        sz[mode] = size(A[mode], 1)
        X = fold(Xmat, mode, sz)
        t3 = time_ns()

        t_reorg += (t1 - t0 + t3 - t2) / 1e6
        t_comp  += (t2 - t1) / 1e6
    end

    return X, SplitTimes(t_plan, t_comp, t_reorg)
end

function NonNaiveMultiplication_timed(X::AbstractArray, A::Vector{<:AbstractMatrix}, P::Vector{Int})
    t0 = time_ns()
    order = OptimalOrdering(X, A)
    t1 = time_ns()

    Y, s = NaiveMultiplication_timed(X, A, order)
    return Y, SplitTimes((t1 - t0) / 1e6 + s.planning_ms, s.compute_ms, s.reorg_ms)
end

function CyclicShiftMultiplication_timed(X::AbstractArray, A::Vector{<:AbstractMatrix}, P::Vector{Int})
    t_plan = 0.0
    t_comp = 0.0
    t_reorg = 0.0

    m = length(A)
    dims = collect(size(X))

    t0 = time_ns()
    X = reshape(X, dims[1], :)
    R = insert!(size.(A, 1), 1, 1)
    t1 = time_ns()
    t_reorg += (t1 - t0) / 1e6

    a = 1
    for i in 1:m
        tp0 = time_ns()
        a *= R[i]
        tp1 = time_ns()
        t_plan += (tp1 - tp0) / 1e6

        tc0 = time_ns()
        X = transpose(X) * transpose(A[i])
        tc1 = time_ns()
        t_comp += (tc1 - tc0) / 1e6

        tr0 = time_ns()
        tailprod = isempty(dims[i+2:end]) ? 1 : prod(dims[i+2:end])
        X = reshape(X, div(size(X, 1), tailprod * a), :)
        tr1 = time_ns()
        t_reorg += (tr1 - tr0) / 1e6
    end

    tr0 = time_ns()
    deleteat!(R, 1)
    X = reshape(X, vcat(R, dims[m+1:end])...)
    tr1 = time_ns()
    t_reorg += (tr1 - tr0) / 1e6

    return X, SplitTimes(t_plan, t_comp, t_reorg)
end

# ============================================================
# 4. Method registry
# ============================================================
const METHOD_ORDER = [:Naïve, :Ordered, :CyclicShift]
const METHODS = Dict{Symbol,Function}(
    :Naïve       => NaiveMultiplication_timed,
    :Ordered     => NonNaiveMultiplication_timed,
    :CyclicShift => CyclicShiftMultiplication_timed,
)

# ============================================================
# 5. Case generation
# ============================================================

const CONSTANT_N = 2^4 * 5^9 # = 

function make_case_constantMemory(order::Symbol, d::Int; N::Int = CONSTANT_N)
    dims = collect(factor_based_shape(N, d))
    X = randn(dims...)

    sizes = order === :normal  ? copy(dims) :
            order === :shuffle ? shuffle(copy(dims)) :
            order === :reverse ? reverse(copy(dims)) :
            error("Unknown order: $order")

    @printf("d = %d | dims = %s | X = %.2f MB\n",
            d, string(Tuple(dims)), sizeof(X) / 1e6)

    A = [randn(sizes[(i)], dims[i]) for i in 1:d]
    return X, A
end

function make_case(order::Symbol, n::Int, d::Int)
    X = randn(ntuple(_ -> n, d))
    idx = collect(1:d)

    sizes = order === :normal  ? idx .* n :
            order === :shuffle ? shuffle(idx) .* n :
            order === :reverse ? reverse(idx) .* n :
            error("Unknown order: $order")

    @printf("d = %d | sizes = %s\n", d, string(sizes))

    A = [randn(s, n) for s in sizes]
    return X, A
end

# ============================================================
# 6. Benchmark
# ============================================================

function benchmark_case(order_sym::Symbol, n::Int, d::Int;
                        methods = METHODS,
                        constant_memory::Bool = false,
                        nreps::Int = 5)

    X, A = constant_memory ? make_case_constantMemory(order_sym, d) : make_case(order_sym, n, d)
    P = collect(1:d)

    results = Dict{Symbol,SplitTimes}()

    for (name, f) in methods
        f(X, A, P)  # warm-up

        plans = Float64[]
        comps = Float64[]
        reorgs = Float64[]

        for _ in 1:nreps
            _, s = f(X, A, P)
            push!(plans, s.planning_ms)
            push!(comps, s.compute_ms)
            push!(reorgs, s.reorg_ms)
        end

        results[name] = SplitTimes(median(plans), median(comps), median(reorgs))

        s = results[name]
        @printf("%-12s | planning = %8.3f ms | compute = %8.3f ms | reorg = %8.3f ms | total = %8.3f ms\n",
                String(name), s.planning_ms, s.compute_ms, s.reorg_ms,
                s.planning_ms + s.compute_ms + s.reorg_ms)
    end

    return results
end

function run_experiments(n::Int, dims::AbstractVector{<:Int};
                         methods = METHODS,
                         constant_memory::Bool = false,
                         nreps::Int = 5)

    method_keys = collect(keys(methods))
    stats = Dict(m => SplitTimes[] for m in method_keys)

    for d in dims
        println("===== d = $d =====")
        case_results = benchmark_case(:normal, n, d;
                                      methods=methods,
                                      constant_memory=constant_memory,
                                      nreps=nreps)
        for m in method_keys
            push!(stats[m], case_results[m])
        end
    end

    return BenchmarkResults(collect(dims), method_keys, stats)
end

# ============================================================
# 7. Plot
# ============================================================

const COL_COMP = RGB(0.11, 0.62, 0.46)
const COL_REOR = RGB(0.53, 0.53, 0.50)

function make_plot(res::BenchmarkResults)
    methods = METHOD_ORDER
    dims = res.dims
    nd = length(dims)
    nm = length(methods)

    comp = zeros(nm, nd)
    reor = zeros(nm, nd)

    for (mi, m) in enumerate(methods), di in 1:nd
        s = res.stats[m][di]
        comp[mi, di] = s.compute_ms
        reor[mi, di] = s.reorg_ms
    end

    bar_w = 0.75 / nm

    p = plot(
        xlabel = "Tensororde d",
        ylabel = "Mediane tijd (ms)",
        title = "Multilineaire vermenigvuldiging — timing per fase",
        legend = :topleft,
        size = (800, 500),
        xticks = (1:nd, string.(dims)),
    )

    # vaste visuele markering per algoritme
    method_markers = Dict(
        :Naïve       => :circle,
        :Ordered     => :square,
        :CyclicShift => :diamond,
    )

    method_labels = Dict(
        :Naïve       => "Naïve",
        :Ordered     => "Ordered",
        :CyclicShift => "Cyclic Shift",
    )

    for (mi, m) in enumerate(methods)
    xs = collect(1:nd) .+ (mi - (nm + 1) / 2) * bar_w
    totals = reor[mi, :] .+ comp[mi, :]

    # Draw the FULL bar first (this becomes the top/arithmetic segment visually)
    bar!(p, xs, totals;
         bar_width = bar_w * 0.9,
         fillcolor = COL_COMP,
         linecolor = :black,
         linewidth = 1,
         label = mi == 1 ? "Arithmetic" : "")

    # Draw the reorg bar on top of it — it covers the bottom part of the total,
    # so the visible stack is: memory movement (bottom) + arithmetic (top)
    bar!(p, xs, reor[mi, :];
         bar_width = bar_w * 0.9,
         fillcolor = :blue,
         linecolor = :black,
         linewidth = 1,
         label = mi == 1 ? "Memory movement" : "")

    # Marker sits right at the top of the stacked bar
    scatter!(p, xs, totals;
             marker = method_markers[m],
             markersize = 7,
             markerstrokecolor = :black,
             markercolor = :white,
             label = method_labels[m])
end
    return p
end

# ============================================================
# 8. Main
# ============================================================

function main(; n::Int = 2,
               dims = 3:13,
               seed::Int = 1234,
               constant_memory::Bool = false,
               nreps::Int = 5,
               savepath::String = "split_timing.png")
    Random.seed!(seed)

    res = run_experiments(n, collect(dims);
                          constant_memory=constant_memory,
                          nreps=nreps)

    fig = make_plot(res)
    display(fig)
    savefig(fig, savepath)
    @printf("\nPlot saved to %s\n", savepath)

    return res
end

main(constant_memory=true)