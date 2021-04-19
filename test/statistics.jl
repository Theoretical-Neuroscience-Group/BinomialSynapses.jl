@testset "statistics" begin
    @testset "MAP" begin
        model = BinomialModel(
                    [1, 2, 2, 2, 4, 4],
                    [0.1, 0.3, 0.4, 0.5, 0.5, 0.5],
                    [0.3, 0.3, 0.7, 0.1, 0.2, 0.3],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    [0.1, 0.1, 0.2, 0.3, 0.5, 0.5]
        )

        map = MAP(model)

        @test map[:N] == 2
        @test map[:p] == 0.5
        @test map[:q] == 0.3
        @test map[:σ] == 0.1
        @test map[:τ] == 0.1
    end
end
