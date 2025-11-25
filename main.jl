using TensorToolbox
using BenchmarkTools
using Random
using Plots
using Statistics
using Printf
using LinearAlgebra

include("naive.jl")
include("orderingMultiplication.jl")    

function make_case_normal(n::Int, d::Int)
    X = randn(ntuple(_ -> n, d))
    sizes = (collect(1:d)) .* n  # smalles first
    @printf "sizes = %s\n" string(sizes)
    A = MatrixCell([randn(s, n) for s in sizes])
    return X, A
end

function make_case_shuffle(n::Int, d::Int)
    X = randn(ntuple(_ -> n, d))
    sizes = shuffle(collect(1:d)) .* n  # random order
    @printf "sizes = %s\n" string(sizes)
    A = MatrixCell([randn(s, n) for s in sizes])
    return X, A
end
function make_case_reverse(n::Int, d::Int)
    X = randn(ntuple(_ -> n, d))
    sizes = reverse(collect(1:d)) .* n  # largest first
    @printf "sizes = %s\n" string(sizes)
    A = MatrixCell([randn(s, n) for s in sizes])
    return X, A
end

# parameters
n = 3
dims = 5:7
medians1_normal = Float64[]
medians2_normal = Float64[]
medians1_shuffle = Float64[]
medians2_shuffle = Float64[]
medians1_reverse = Float64[]
medians2_reverse = Float64[]
mem1_normal = Float64[]   
mem2_normal = Float64[]   
mem1_shuffle = Float64[]   
mem2_shuffle = Float64[]   
mem1_reverse = Float64[]   
mem2_reverse = Float64[]   
allocs1_normal = Int[]
allocs2_normal = Int[]
allocs1_shuffle = Int[]
allocs2_shuffle = Int[]
allocs1_reverse = Int[]
allocs2_reverse = Int[]

for d in dims
    X_normal, A_normal = make_case_normal(n, d)
    X_shuffle, A_shuffle = make_case_shuffle(n, d)
    X_reverse, A_reverse = make_case_reverse(n, d)

    t1_normal = @benchmark NaiveMultiplication($X_normal, $A_normal) #@time voor debuggen  run altijd minstens 2 keer want JIT
    t2_normal = @benchmark NonNaiveMultiplication($X_normal, $A_normal)
    t1_shuffle = @benchmark NaiveMultiplication($X_shuffle, $A_shuffle)
    t2_shuffle = @benchmark NonNaiveMultiplication($X_shuffle, $A_shuffle)
    t1_reverse = @benchmark NaiveMultiplication($X_reverse, $A_reverse)
    t2_reverse = @benchmark NonNaiveMultiplication($X_reverse, $A_reverse)



    m1 = median(t1_normal.times) / 1e6   # ms
    m2 = median(t2_normal.times) / 1e6
    push!(medians1_normal, m1)
    push!(medians2_normal, m2)
    push!(mem1_normal, t1_normal.memory)        # bytes per evaluation
    push!(mem2_normal, t2_normal.memory)
    push!(allocs1_normal, t1_normal.allocs)
    push!(allocs2_normal, t2_normal.allocs)
    push!(medians1_shuffle, median(t1_shuffle.times) / 1e6)
    push!(medians2_shuffle, median(t2_shuffle.times) / 1e6)
    push!(mem1_shuffle, t1_shuffle.memory)
    push!(mem2_shuffle, t2_shuffle.memory)
    push!(allocs1_shuffle, t1_shuffle.allocs)
    push!(allocs2_shuffle, t2_shuffle.allocs)
    push!(medians1_reverse, median(t1_reverse.times) / 1e6)
    push!(medians2_reverse, median(t2_reverse.times) / 1e6)
    push!(mem1_reverse, t1_reverse.memory)
    push!(mem2_reverse, t2_reverse.memory)
    push!(allocs1_reverse, t1_reverse.allocs)
    push!(allocs2_reverse, t2_reverse.allocs)
    @printf "d = %d | time Naive = %.3f ms, ordering = %.3f ms | mem Naive = %.2f MB, ordering = %.2f MB | allocs Naive = %d, ordering = %d\n" d m1 m2 (t1_normal.memory/1e6) (t2_normal.memory/1e6) t1_normal.allocs t2_normal.allocs
end

# Plots: top = time, bottom = memory
p_time = plot(
    dims, medians1_normal,
    label = "Naive time normal ordering",
    xlabel = "Tensor Order (d)",
    ylabel = "Median Time (ms)",
    title = "Multilinear Multiplication: Time vs Memory",
    legend = :topleft
)
plot!(p_time, dims, medians1_shuffle, label = "Naive time shuffle ordering")
plot!(p_time, dims, medians1_reverse, label = "Naive time reverse ordering")
plot!(p_time, dims, medians2_normal, label = "Ordering time normal ordering")
plot!(p_time, dims, medians2_shuffle, label = "Ordering time shuffle ordering")
plot!(p_time, dims, medians2_reverse, label = "Ordering time reverse ordering")


p_mem = plot(
    dims, mem1_normal ./ 1e6,
    label = "Naive memory normal ordering",
    xlabel = "Tensor Order (d)",
    ylabel = "Memory (MB)",
    legend = :topleft
)
plot!(p_mem, dims, mem1_shuffle ./ 1e6, label = "Naive memory shuffle ordering")
plot!(p_mem, dims, mem1_reverse ./ 1e6, label = "Naive memory reverse ordering")
plot!(p_mem, dims, mem2_normal ./ 1e6, label = "Ordering memory normal ordering")
plot!(p_mem, dims, mem2_shuffle ./ 1e6, label = "Ordering memory shuffle ordering")
plot!(p_mem, dims, mem2_reverse ./ 1e6, label = "Ordering memory reverse ordering")

plot(p_time, p_mem, layout = (2, 1), size = (900, 700))
