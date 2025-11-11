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
dims = 1:12
medians1 = Float64[]
medians2 = Float64[]

for d in dims
    X, A = make_case(n, d)
    t1 = @benchmark NaiveMultiplication($X, $A)
    t2 = @benchmark NonNaiveMultiplication($X, $A)
    m1 = median(t1.times) / 1e6   # convert to milliseconds
    m2 = median(t2.times) / 1e6
    push!(medians1, m1)
    push!(medians2, m2)
    @printf "d = %d, median Naive = %.3f ms, median ordering = %.3f ms\n" d m1 m2
end

#plot 
plot(dims, mediansNaive, label="Naive Multiplication", xlabel="Tensor Order (d)", ylabel="Median Time (ms)", title="Performance Comparison of Multilinear Multiplication Methods", legend=:topleft)
plot!(dims, mediansTTM, label="ordering Multiplication")
