using Test, Random
using TensorToolbox
using Strided

include("../algorithms/cyclicShift.jl")
include("../algorithms/orderingMultiplication.jl")

# ── Reference via TensorToolbox ───────────────────────────────────────────────
function ttm_reference(X::AbstractArray, A::Vector{<:AbstractMatrix})
    result = X
    for (i, Ai) in enumerate(A)
        result = ttm(result, Ai, i)
    end
    return result
end

Random.seed!(42)

@testset "Cyclic shift strategies vs TensorToolbox.ttm" begin

    @testset "m == ndims == 2" begin
        X = randn(3, 4)
        A = [randn(5, 3), randn(6, 4)]
        active = [1, 2]

        ref = ttm_reference(X, A)

        @test multiply_with_unity_fill(X, A, active) ≈ ref
        @test multiply_with_permutation(X, A, active) ≈ ref
    end

    @testset "m == ndims == 3" begin
        X = randn(3, 4, 5)
        A = [randn(6, 3), randn(7, 4), randn(8, 5)]
        active = [1, 2, 3]

        ref = ttm_reference(X, A)

        @test multiply_with_unity_fill(X, A, active) ≈ ref
        @test multiply_with_permutation(X, A, active) ≈ ref
    end

    @testset "m=1 < ndims=2" begin
        X = randn(3, 4)
        A = [randn(5, 3)]
        active = [1]

        ref = ttm_reference(X, A)

        @test multiply_with_unity_fill(X, A, active) ≈ ref
        @test multiply_with_permutation(X, A, active) ≈ ref
    end

    @testset "m=1 < ndims=3" begin
        X = randn(3, 4, 5)
        A = [randn(6, 3)]
        active = [1]

        ref = ttm_reference(X, A)

        @test multiply_with_unity_fill(X, A, active) ≈ ref
        @test multiply_with_permutation(X, A, active) ≈ ref
    end

    @testset "m=2 < ndims=3" begin
        X = randn(3, 4, 5)
        A = [randn(6, 3), randn(7, 4)]
        active = [1, 2]

        ref = ttm_reference(X, A)

        @test multiply_with_unity_fill(X, A, active) ≈ ref
        @test multiply_with_permutation(X, A, active) ≈ ref
    end

    @testset "m=1 < ndims=4" begin
        X = randn(3, 4, 5, 6)
        A = [randn(7, 3)]
        active = [1]

        ref = ttm_reference(X, A)

        @test multiply_with_unity_fill(X, A, active) ≈ ref
        @test multiply_with_permutation(X, A, active) ≈ ref
    end

    @testset "m=3 < ndims=4" begin
        X = randn(3, 4, 5, 6)
        A = [randn(7, 3), randn(8, 4), randn(9, 5)]
        active = [1, 2, 3]

        ref = ttm_reference(X, A)

        @test multiply_with_unity_fill(X, A, active) ≈ ref
        @test multiply_with_permutation(X, A, active) ≈ ref
    end

end