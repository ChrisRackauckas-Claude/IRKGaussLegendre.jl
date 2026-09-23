using Test
using IRKGaussLegendre, SciMLBase

function lotka_volterra!(du, u, p, t)
    du[1] = 1.5 * u[1] - u[1] * u[2]
    du[2] = -3 * u[2] + u[1] * u[2]
    return nothing
end

maxdiff(a, b) = maximum(maximum(abs.(x .- y)) for (x, y) in zip(a, b))

@testset "Dense output uses the collocation interpolant" begin
    prob = ODEProblem(lotka_volterra!, [1.0, 1.0], (0.0, 10.0))
    ts = 0.05:0.1:9.95
    ref = solve(prob, IRKGL16(), abstol = 1.0e-15, reltol = 1.0e-15, saveat = ts)

    @testset "simd = $simd" for simd in (false, true)
        sol = solve(prob, IRKGL16(simd = simd), abstol = 1.0e-12, reltol = 1.0e-12)
        sa = solve(prob, IRKGL16(simd = simd), abstol = 1.0e-12, reltol = 1.0e-12, saveat = ts)
        @test sol.dense
        @test maxdiff(sol.(ts), ref.u[2:(end - 1)]) < 1.0e-6
        @test maxdiff(sol.(ts), sa.u[2:(end - 1)]) < 1.0e-12
        @test maxdiff(sol(ts).u, sol.(ts)) == 0
        @test all(sol(sol.t[i]) == sol.u[i] for i in eachindex(sol.t))
        @test sol(3.3; idxs = 2) == sol(3.3)[2]
        @test sol(3.3; idxs = [2, 1]) == sol(3.3)[[2, 1]]
        @test_throws ErrorException sol(10.5)
    end

    @testset "backward integration" begin
        bprob = ODEProblem(lotka_volterra!, ref.u[end], (10.0, 0.0))
        sol = solve(bprob, IRKGL16(), abstol = 1.0e-12, reltol = 1.0e-12)
        @test maxdiff(sol.(ts), ref.u[2:(end - 1)]) < 1.0e-6
    end

    @testset "fixed step" begin
        sol = solve(prob, IRKGL16(), dt = 0.25, adaptive = false)
        sa = solve(prob, IRKGL16(), dt = 0.25, adaptive = false, saveat = ts)
        @test maxdiff(sol.(ts), sa.u[2:(end - 1)]) < 1.0e-12
        @test maxdiff(sol.(ts), ref.u[2:(end - 1)]) < 1.0e-6
    end

    @testset "BigFloat" begin
        bprob = ODEProblem(lotka_volterra!, BigFloat[1, 1], (big(0.0), big(10.0)))
        sol = solve(bprob, IRKGL16(), abstol = big(1.0e-20), reltol = big(1.0e-20))
        @test eltype(sol(big(3.3))) == BigFloat
        @test maxdiff(sol.(big.(ts)), ref.u[2:(end - 1)]) < 1.0e-6
    end

    @testset "no dense output when not saving every step" begin
        sol = solve(prob, IRKGL16(), save_everystep = false)
        @test !sol.dense
        sa = solve(prob, IRKGL16(), saveat = ts)
        @test !sa.dense
    end
end
