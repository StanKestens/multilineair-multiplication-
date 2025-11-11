using TensorToolbox
using BenchmarkTools
using Random
using Plots
using Statistics
using Printf
using LinearAlgebra

include("naive.jl")

function make_case(n::Int, d::Int)
    X = randn(ntuple(_ -> n, d))
    A = MatrixCell([randn(n, n) for _ in 1:d])
    return X, A
end

# parameters
n = 5
dims = 1:5
medians = Float64[]

for d in dims
    X, A = make_case(n, d)
    t = @benchmark ttm($X, $A)
    m = median(t.times) / 1e6   # convert to milliseconds
    push!(medians, m)
    @printf "d = %d, median = %.3f ms\n" d m
end

# plot
plot(dims, medians, marker=:o, xlabel="Tensor Order (d)", ylabel="Median Time (ms)",
     title="TTM Benchmark vs Tensor Order", legend=false)
