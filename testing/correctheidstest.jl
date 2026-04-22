using BenchmarkTools
using Random
using Plots
using Test

include("../utility/tensor.jl")
include("../algorithms/ordering.jl")
include("../algorithms/naive.jl")
include("../algorithms/cyclicShift.jl")
include("../algorithms/bruteforce.jl")
include("../algorithms/orderingMultiplication.jl")

X = rand(5, 6, 7)

A = [
    randn(2, 5),
    randn(3, 6),
    randn(4, 7)
]

Y = CyclicShiftMultiplication(X, A)

Z = bruteforce(X, A)
B = NaiveMultiplication(X, A, [1, 2, 3])
C = NonNaiveMultiplication(X, A)
@test Y ≈ Z
@test Y ≈ B
@test Y ≈ C

