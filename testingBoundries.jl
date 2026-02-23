using BenchmarkTools
using Random
using Plots

include("tensor.jl")
include("ordering.jl")
include("naive.jl")
include("cyclicShift.jl")

function (d::Int)
    #Setting up a random tensor dimension d
    dimensies = rand(2:10, d) #TODO : make 10 higher when doing actual running 
    X = randn(dimensies...)
    #Generating all A_i
    m = 10 #TODO : deze bound ook hoger maken
    A = [randn(rand(1:m), dimensies[i]) for i in 1:d]
    for i in 1:d # Looping over where the gap is, in more general case this would have to be d choose g options (g amount of gaps)
        #The options
        A_new = A
        #1) running with unity matrix at position i
        setindex!(A_new, I_n=Matrix(I, n, n), i)
        X_unity = CyclicShiftMultiplication(X, A_new, collect(1:d))
        #2) permuting
        #this option is a bit trickier, we permutedim to put our gap at the end
        P = vcat(1:i-1, i+1:d, i) # permutationvector
        X_perm = permutedims(X, P)
        A_perm = A[P]
        X_perm = CyclicShiftMultiplication(X_perm, A_perm, collect(1:d))
        X_perm_back = permutedims(X_perm, invperm(P))
        #these have to be compared in speed
    end
end