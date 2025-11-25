using TensorToolbox
using BenchmarkTools
using Random
using Plots
using Statistics
using Printf
using LinearAlgebra

include("naive.jl")
include("orderingMultiplication.jl")

function make_case(n::Int, d::Int)
    X = randn(ntuple(_ -> n, d))
    sizes = reverse(collect(1:d)) .* n  # largest first
    A = MatrixCell([randn(s, n) for s in sizes])
    return X, A
end

# parameters
n = 5
dims = 1:5
medians1 = Float64[]
medians2 = Float64[]
mem1 = Float64[]   # bytes
mem2 = Float64[]   # bytes
allocs1 = Int[]
allocs2 = Int[]

for d in dims
    X, A = make_case(n, d)

    t1 = @benchmark NaiveMultiplication($X, $A) #@time voor debuggen  run altijd minstens 2 keer want JIT
    t2 = @benchmark NonNaiveMultiplication($X, $A)

    m1 = median(t1.times) / 1e6   # ms
    m2 = median(t2.times) / 1e6
    push!(medians1, m1)
    push!(medians2, m2)

    push!(mem1, t1.memory)        # bytes per evaluation
    push!(mem2, t2.memory)
    push!(allocs1, t1.allocs)
    push!(allocs2, t2.allocs)

    @printf "d = %d | time Naive = %.3f ms, ordering = %.3f ms | mem Naive = %.2f MB, ordering = %.2f MB | allocs Naive = %d, ordering = %d\n" d m1 m2 (t1.memory/1e6) (t2.memory/1e6) t1.allocs t2.allocs
end

# Plots: top = time, bottom = memory
p_time = plot(
    dims, medians1,
    label = "Naive time",
    xlabel = "Tensor Order (d)",
    ylabel = "Median Time (ms)",
    title = "Multilinear Multiplication: Time vs Memory",
    legend = :topleft
)
plot!(p_time, dims, medians2, label = "Ordering time")

p_mem = plot(
    dims, mem1 ./ 1e6,
    label = "Naive memory",
    xlabel = "Tensor Order (d)",
    ylabel = "Memory (MB)",
    legend = :topleft
)
plot!(p_mem, dims, mem2 ./ 1e6, label = "Ordering memory")

plot(p_time, p_mem, layout = (2, 1), size = (900, 700))
