using BenchmarkTools
using Random
using Plots

include("tensor.jl")
include("ordering.jl")
include("naive.jl")
include("cyclicShift.jl")
include("bruteforce.jl")

X = rand(5, 6, 7)

A = [
    randn(2, 5),
    randn(3, 6),
    randn(4, 7)
]

Y = CyclicShiftMultiplication(X, A, [1, 2, 3])

Z = bruteforce(X, A)
B = NaiveMultiplication(X, A, [1, 2, 3])
C = NonNaiveMultiplication(X, A)
@test Y = Z
@test Y = B
@test Y = C

