using BenchmarkTools
using Random
using Plots

include("tensor.jl")
include("ordering.jl")
include("naive.jl")
include("cyclicShift.jl")

X = rand(5, 6, 7)

A = [
    randn(2, 5),
    randn(3, 6)
]

Y = CyclicShiftMultiplication(X, A, [1, 2, 3])

