CUDA.functional() && @testset "filter.jl" begin
    @info "Testing filter.jl"
    @testset "Device = $device" for device in DEVICES
        m_out  = 1024
        m_in   = 1024

        state = BinomialState(128, m_out, m_in, device = device)
        model = BinomialGridModel(
            m_out,
            1:5,
            LinRange(0.05,0.95,5),
            LinRange(0.1,2,5),
            LinRange(0.05,2,5),
            LinRange(0.05,2,5),
            device = device
        )

        fstate = NestedParticleState(state, model)
        filter = NestedParticleFilter(12)
        obs    = BinomialObservation(0.3f0, 0.1f0)

        if device === :gpu
            println("")
            @info "Benchmarking one filter update step: should take less than 10ms."
            display(@benchmark CUDA.@sync update!($fstate, $obs, $filter))
            println("")
            println("")
        end
    end
end
