using TensorToolbox
using BenchmarkTools
using Random
using Plots
using Statistics
using Printf
using LinearAlgebra

include("naive.jl")                  # NaiveMultiplication(X, A, order)
include("orderingMultiplication.jl") # NonNaiveMultiplication(X, A, order)
include("cyclicShift.jl")            # CyclicShiftMultiplication(X, A, order)
include("tensor.jl")
include("ordering.jl")
include("bruteforce.jl")             # bruteforce(X, A)
include("constantMemoryTest.jl")    # constant_memory_shape(d; N, order)

# ============================================================
# 1. Methoden definiëren
# ============================================================

const METHODS = Dict{Symbol,Function}(
    :Naïve       => NaiveMultiplication,
    :Ordered     => (X, A, P) -> NonNaiveMultiplication(X, A),
    :CyclicShift => CyclicShiftMultiplication,
    #:Bruteforce  => (X, A, P) -> bruteforce(X, A),
)

# ============================================================
# 2. Cases genereren
# ============================================================

function make_case(order::Symbol, n::Int, d::Int)
    X = randn(ntuple(_ -> n, d))

    idx = collect(1:d)
    sizes = if order === :normal
        idx .* n
    elseif order === :shuffle
        shuffle(idx) .* n
    elseif order === :reverse
        reverse(idx) .* n
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
# 2b. Constant-memory shapes
# ============================================================

const CONSTANT_N = 1250000 # = 2^5 * 5^7



"""
    constant_memory_shape(d; N=CONSTANT_N, order=:balanced)

Geeft een shape van orde d met exact product N.
"""


"""
    make_case_constantMemory(order, d; N=CONSTANT_N)

Genereer een tensor met constant aantal elementen N maar variërende shape/orde.
De matrixgroottes blijven variëren volgens dezelfde order-logica.
"""
function make_case_constantMemory(order::Symbol, d::Int; N::Int = CONSTANT_N)
    dims = collect(factor_based_shape(N, d))
    X = randn(dims...)

    sizes = if order === :normal
        copy(dims)
    elseif order === :shuffle
        shuffle(copy(dims))
    elseif order === :reverse
        reverse(copy(dims))
    else
        error("Onbekende order: $order")
    end

    @printf(
        "d = %d | order = %s | dims = %s | prod = %d | X = %.2f MB | sizes = %s\n",
        d, String(order), string(Tuple(dims)), prod(dims), sizeof(X)/1e6, string(sizes)
    )

    A = MatrixCell([randn(sizes[i], dims[i]) for i in 1:d])

    return X, A
end

# ============================================================
# 3. Structs voor resultaten
# ============================================================

struct MethodStats
    time_ms::Float64
    memory::Float64
    allocs::Int
end

struct BenchmarkResults
    dims::Vector{Int}
    orders::Vector{Symbol}
    methods::Vector{Symbol}
    times::Dict{Symbol,Dict{Symbol,Vector{Float64}}}
    mem::Dict{Symbol,Dict{Symbol,Vector{Float64}}}
    allocs::Dict{Symbol,Dict{Symbol,Vector{Int}}}
end

# ============================================================
# 4. Benchmark voor één case
# ============================================================

function benchmark_case(order_sym::Symbol, n::Int, d::Int; methods = METHODS, constant_memory::Bool = false)
    X, A = constant_memory ? make_case_constantMemory(order_sym, d) : make_case(order_sym, n, d)

    P = collect(1:d)
    results = Dict{Symbol,MethodStats}()

    println("  Methoden voor order = $(order_sym):")
    for (mname, f) in methods
        t = @benchmark $f($X, $A, $P)
        time_ms = median(t.times) / 1e6
        mem     = t.memory
        allocs  = t.allocs

        @printf(
            "    %-12s: time = %.3f ms | mem = %.2f MB | allocs = %d\n",
            String(mname), time_ms, mem / 1e6, allocs
        )

        results[mname] = MethodStats(time_ms, mem, allocs)
    end

    return results
end

# ============================================================
# 5. Alle experimenten draaien
# ============================================================

function run_experiments(n::Int, dims::AbstractVector{<:Int}; methods = METHODS, constant_memory::Bool = false)
    orders = [:normal]

    times  = Dict(m => Dict(o => Float64[] for o in orders) for m in keys(methods))
    mems   = Dict(m => Dict(o => Float64[] for o in orders) for m in keys(methods))
    allocs = Dict(m => Dict(o => Int[]     for o in orders) for m in keys(methods))

    for d in dims
        println("===== d = $d =====")
        for o in orders
            case_results = benchmark_case(o, n, d; methods = methods, constant_memory = constant_memory)
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

function make_plots(res::BenchmarkResults)
    avg_times, avg_mem = average_over_orders(res)

    p_time = plot(
        xlabel = "Tensor Order (d)",
        ylabel = "Average Median Time (ms)",
        title  = "Multilinear Multiplication: Average Time",
        legend = :topleft,
        yscale = :log10,
    )

    for m in res.methods
        plot!(
            p_time,
            res.dims,
            avg_times[m],
            label = string(m),
            lw = 2,
            marker = :circle,
            markersize = 5,
            markerstrokewidth = 1,
        )
    end

    p_mem = plot(
        xlabel = "Tensor Order (d)",
        ylabel = "Average Memory (MB)",
        title  = "Multilinear Multiplication: Average Memory",
        legend = :topleft,
        yscale = :log10,
    )

    for m in res.methods
        plot!(
            p_mem,
            res.dims,
            avg_mem[m] ./ 1e6,
            label = string(m),
            lw = 2,
            marker = :circle,
            markersize = 5,
            markerstrokewidth = 1,
        )
    end

    return plot(p_time, p_mem, layout = (2, 1), size = (900, 700))
end

# ============================================================
# 7. main
# ============================================================

function main(; n::Int = 2, dims = 3:10, seed::Int = 1234, methods = METHODS, constant_memory::Bool = false)
    Random.seed!(seed)
    res = run_experiments(n, collect(dims); methods = methods, constant_memory = constant_memory)
    fig = make_plots(res)
    display(fig)
    return res
end

# Standaard experiment
# main()

# Constant-memory experiment
main(constant_memory = true)