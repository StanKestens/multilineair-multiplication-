using TensorToolbox
using BenchmarkTools
using Random
using Plots
using Statistics
using Printf
using LinearAlgebra

include("naive.jl")                  # NaiveMultiplication(X, A, order)
include("orderingMultiplication.jl") # NonNaiveMultiplication(X, A, order)
include("cyclicShift.jl")           # CyclicShiftMultiplication(X, A, order)
include("tensor.jl")
include("ordering.jl")

# ============================================================
# 1. Methoden definiëren (HIER voeg je nieuwe functies toe)
# ============================================================

# Alle methodes hebben signatuur: f(X::AbstractArray, A::MatrixCell, order::Vector{Int})
const METHODS = Dict{Symbol,Function}(
    :Naive   => NaiveMultiplication,
    :Ordered => NonNaiveMultiplication,
    :CyclicShift => CyclicShiftMultiplication,   # voorbeeld
    # :NewAlg => NewAlgMultiplication,   # voorbeeld
)

# ============================================================
# 2. Cases genereren
# ============================================================

function make_case(order::Symbol, n::Int, d::Int)
    X = randn(ntuple(_ -> n, d))

    idx = collect(1:d)
    sizes = if order === :normal
        idx .* n                     # kleinste eerst
    elseif order === :shuffle
        shuffle(idx) .* n            # willekeurige volgorde
    elseif order === :reverse
        reverse(idx) .* n            # grootste eerst
    else
        error("Onbekende order: $order")
    end

    @printf(
        "d = %d | order = %s | sizes = %s\n",
        d, String(order), string(sizes)
    )

    A = MatrixCell([randn(s, n) for s in sizes])
    return X, A
end

# ============================================================
# 3. Structs voor resultaten
# ============================================================

struct MethodStats
    time_ms::Float64    # mediane tijd in ms
    memory::Float64     # bytes
    allocs::Int         # aantal allocaties
end

struct BenchmarkResults
    dims::Vector{Int}
    orders::Vector{Symbol}
    methods::Vector{Symbol}
    times::Dict{Symbol,Dict{Symbol,Vector{Float64}}}   # method => order(sym) => [tijd]
    mem::Dict{Symbol,Dict{Symbol,Vector{Float64}}}     # method => order(sym) => [bytes]
    allocs::Dict{Symbol,Dict{Symbol,Vector{Int}}}      # method => order(sym) => [allocs]
end

# ============================================================
# 4. Benchmark voor één (order_sym, n, d)
# ============================================================

function benchmark_case(order_sym::Symbol, n::Int, d::Int; methods = METHODS)
    X, A = make_case(order_sym, n, d)

    # mode order voor alle methodes (pas dit aan als je andere permutaties wil testen)
    P = collect(1:d)

    results = Dict{Symbol,MethodStats}()

    println("  Methoden voor order = $(order_sym):")
    for (mname, f) in methods
        t = @benchmark $f($X, $A, $P)
        time_ms = median(t.times) / 1e6
        mem     = t.memory
        allocs  = t.allocs

        @printf(
            "    %-10s: time = %.3f ms | mem = %.2f MB | allocs = %d\n",
            String(mname), time_ms, mem/1e6, allocs
        )

        results[mname] = MethodStats(time_ms, mem, allocs)
    end

    return results
end

# ============================================================
# 5. Alle experimenten draaien
# ============================================================

function run_experiments(n::Int, dims::AbstractVector{<:Int}; methods = METHODS)
    orders = [:normal, :shuffle, :reverse]

    # method => order(sym) => vector
    times  = Dict(m => Dict(o => Float64[] for o in orders) for m in keys(methods))
    mems   = Dict(m => Dict(o => Float64[] for o in orders) for m in keys(methods))
    allocs = Dict(m => Dict(o => Int[]     for o in orders) for m in keys(methods))

    for d in dims
        println("===== d = $d =====")
        for o in orders
            case_results = benchmark_case(o, n, d; methods = methods)
            for (mname, stats) in case_results
                push!(times[mname][o],  stats.time_ms)
                push!(mems[mname][o],   stats.memory)
                push!(allocs[mname][o], stats.allocs)
            end
        end
    end

    return BenchmarkResults(
        collect(dims),
        orders,
        collect(keys(methods)),
        times,
        mems,
        allocs,
    )
end

# ============================================================
# 6. Plotten
# ============================================================
function make_plots(res::BenchmarkResults)

    avg_times, avg_mem = average_over_orders(res)

    # Tijd
    p_time = plot(
        xlabel = "Tensor Order (d)",
        ylabel = "Average Median Time (ms)",
        title  = "Multilinear Multiplication: Average Time",
        legend = :topleft,
        yscale = :log10,
        
    )

    for m in res.methods
        plot!(p_time, res.dims, avg_times[m], label = string(m))
    end

    # Geheugen
    p_mem = plot(
        xlabel = "Tensor Order (d)",
        ylabel = "Average Memory (MB)",
        title  = "Multilinear Multiplication: Average Memory",
        legend = :topleft,
        yscale = :log10,
        
    )

    for m in res.methods
        plot!(p_mem, res.dims, avg_mem[m] ./ 1e6, label = string(m))
    end

    return plot(p_time, p_mem, layout = (2, 1), size = (900, 700))
end

function average_over_orders(res::BenchmarkResults)
    avg_times = Dict{Symbol,Vector{Float64}}()
    avg_mem   = Dict{Symbol,Vector{Float64}}()

    for m in res.methods
        avg_times[m] = Float64[]
        avg_mem[m]   = Float64[]

        for i in eachindex(res.dims)
            tvals = [res.times[m][o][i] for o in res.orders]
            mvals = [res.mem[m][o][i]   for o in res.orders]

            push!(avg_times[m], mean(tvals))
            push!(avg_mem[m],   mean(mvals))
        end
    end

    return avg_times, avg_mem
end
# ============================================================
# 7. main
# ============================================================

#voor dims 3:6
function main(; n::Int = 2, dims = 5:7, seed::Int = 1234, methods = METHODS)
    Random.seed!(seed)
    res = run_experiments(n, collect(dims); methods = methods)
    fig = make_plots(res)
    display(fig)
    return res
end

main()
