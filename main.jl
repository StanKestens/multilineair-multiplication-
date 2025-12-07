using TensorToolbox
using BenchmarkTools
using Random
using Plots
using Statistics
using Printf
using LinearAlgebra

include("naive.jl")
include("orderingMultiplication.jl")
include("cyclicShift.jl")
include("tensor.jl")
include("ordering.jl")

# --------------------------
# Cases genereren
# --------------------------

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

# --------------------------
# Benchmark voor één (order, n, d)
# --------------------------

function benchmark_case(order::Symbol, n::Int, d::Int)
    X, A = make_case(order, n, d)

    t1 = @benchmark NaiveMultiplication($X, $A)
    m1   = median(t1.times) / 1e6     # ms
    mem1 = t1.memory                  # bytes
    a1   = t1.allocs
    
    t2 = @benchmark NonNaiveMultiplication($X, $A)
    m2   = median(t2.times) / 1e6
    mem2 = t2.memory
    a2   = t2.allocs
    
    t3 = @benchmark CyclicShiftMultiplication($X, $A, collect(1:length(A)))
    m3   = median(t3.times) / 1e6
    mem3 = t3.memory
    a3   = t3.allocs

    @printf(
        "  -> time Naive = %.3f ms, ordering = %.3f ms, cyclic = %.3f ms | mem Naive = %.2f MB, ordering = %.2f MB, cyclic = %.2f MB | allocs Naive = %d, ordering = %d, cyclic = %d\n",
        m1, m2, m3, mem1/1e6, mem2/1e6, mem3/1e6, a1, a2, a3
    )

    return m1, m2, m3, mem1, mem2, mem3, a1, a2, a3
end

# --------------------------
# Struct voor resultaten
# --------------------------

struct BenchmarkResults
    dims::Vector{Int}
    orders::Vector{Symbol}
    medians1::Dict{Symbol,Vector{Float64}}   # Naive tijd
    medians2::Dict{Symbol,Vector{Float64}}   # NonNaive tijd
    medians3::Dict{Symbol,Vector{Float64}}   # Cyclic tijd
    mem1::Dict{Symbol,Vector{Float64}}       # Naive memory (bytes)
    mem2::Dict{Symbol,Vector{Float64}}       # NonNaive memory (bytes)
    mem3::Dict{Symbol,Vector{Float64}}       # Cyclic memory (bytes)
    allocs1::Dict{Symbol,Vector{Int}}
    allocs2::Dict{Symbol,Vector{Int}}
    allocs3::Dict{Symbol,Vector{Int}}
end

# --------------------------
# Alle experimenten draaien
# --------------------------

function run_experiments(n::Int, dims::AbstractVector{<:Int})
    orders = [:normal, :shuffle, :reverse]

    med1 = Dict(o => Float64[] for o in orders)
    med2 = Dict(o => Float64[] for o in orders)
    med3 = Dict(o => Float64[] for o in orders)
    mem1 = Dict(o => Float64[] for o in orders)
    mem2 = Dict(o => Float64[] for o in orders)
    mem3 = Dict(o => Float64[] for o in orders)
    a1   = Dict(o => Int[]      for o in orders)
    a2   = Dict(o => Int[]      for o in orders)
    a3   = Dict(o => Int[]      for o in orders)

    for d in dims
        println("===== d = $d =====")
        for o in orders
            m1, m2, m3, mm1, mm2, mm3, aa1, aa2, aa3 = benchmark_case(o, n, d)
            push!(med1[o], m1)
            push!(med2[o], m2)
            push!(med3[o], m3)
            push!(mem1[o], mm1)
            push!(mem2[o], mm2)
            push!(mem3[o], mm3)
            push!(a1[o], aa1)
            push!(a2[o], aa2)
            push!(a3[o], aa3)
        end
    end

    return BenchmarkResults(collect(dims), orders, med1, med2, med3, mem1, mem2, mem3, a1, a2, a3)
end

# --------------------------
# Plotten
# --------------------------

function make_plots(res::BenchmarkResults)
    labels_time = Dict(
        :normal  => ("Naive time normal ordering",    "Ordering time normal ordering", "Cyclic time normal ordering"),
        :shuffle => ("Naive time shuffle ordering",   "Ordering time shuffle ordering", "Cyclic time shuffle ordering"),
        :reverse => ("Naive time reverse ordering",   "Ordering time reverse ordering", "Cyclic time reverse ordering"),
    )

    labels_mem = Dict(
        :normal  => ("Naive memory normal ordering",  "Ordering memory normal ordering", "Cyclic memory normal ordering"),
        :shuffle => ("Naive memory shuffle ordering", "Ordering memory shuffle ordering", "Cyclic memory shuffle ordering"),
        :reverse => ("Naive memory reverse ordering", "Ordering memory reverse ordering", "Cyclic memory reverse ordering"),
    )

    # Tijd
    p_time = plot(
        res.dims, res.medians1[:normal],
        label  = labels_time[:normal][1],
        xlabel = "Tensor Order (d)",
        ylabel = "Median Time (ms)",
        title  = "Multilinear Multiplication: Time",
        legend = :topleft,
    )
    plot!(p_time, res.dims, res.medians1[:shuffle], label = labels_time[:shuffle][1])
    plot!(p_time, res.dims, res.medians1[:reverse], label = labels_time[:reverse][1])
    plot!(p_time, res.dims, res.medians2[:normal],  label = labels_time[:normal][2])
    plot!(p_time, res.dims, res.medians2[:shuffle], label = labels_time[:shuffle][2])
    plot!(p_time, res.dims, res.medians2[:reverse], label = labels_time[:reverse][2])
    plot!(p_time, res.dims, res.medians3[:normal],  label = labels_time[:normal][3])
    plot!(p_time, res.dims, res.medians3[:shuffle], label = labels_time[:shuffle][3])
    plot!(p_time, res.dims, res.medians3[:reverse], label = labels_time[:reverse][3])

    # Geheugen (naar MB)
    p_mem = plot(
        res.dims, res.mem1[:normal] ./ 1e6,
        label  = labels_mem[:normal][1],
        xlabel = "Tensor Order (d)",
        ylabel = "Memory (MB)",
        legend = :topleft,
        title  = "Multilinear Multiplication: Memory",
    )
    plot!(p_mem, res.dims, res.mem1[:shuffle] ./ 1e6, label = labels_mem[:shuffle][1])
    plot!(p_mem, res.dims, res.mem1[:reverse] ./ 1e6, label = labels_mem[:reverse][1])
    plot!(p_mem, res.dims, res.mem2[:normal] ./ 1e6,  label = labels_mem[:normal][2])
    plot!(p_mem, res.dims, res.mem2[:shuffle] ./ 1e6, label = labels_mem[:shuffle][2])
    plot!(p_mem, res.dims, res.mem2[:reverse] ./ 1e6, label = labels_mem[:reverse][2])
    plot!(p_mem, res.dims, res.mem3[:normal] ./ 1e6,  label = labels_mem[:normal][3])
    plot!(p_mem, res.dims, res.mem3[:shuffle] ./ 1e6, label = labels_mem[:shuffle][3])
    plot!(p_mem, res.dims, res.mem3[:reverse] ./ 1e6, label = labels_mem[:reverse][3])

    return plot(p_time, p_mem, layout = (2, 1), size = (900, 700))
end

# --------------------------
# main
# --------------------------

function main(; n::Int = 2, dims = 1:2, seed::Int = 1234)
    Random.seed!(seed)
    res = run_experiments(n, collect(dims))
    fig = make_plots(res)
    display(fig)
    return res
end

main()
