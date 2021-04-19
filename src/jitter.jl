using CUDA

function jitter!(model::BinomialGridModel, width)
    prob = Float32(1/width)

    jitter!(model.Nind, length(model.Nrng), prob)
    jitter!(model.pind, length(model.prng), prob)
    jitter!(model.qind, length(model.qrng), prob)
    jitter!(model.σind, length(model.σrng), prob)
    jitter!(model.τind, length(model.τrng), prob)

    refresh!(model)
    return model
end

function jitter!(indices, maxindex, prob)
    function kernel(indices, range, prob, randstates)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        @inbounds if i <= length(indices)
            r   = GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates)
            idx = indices[i]
            if r < prob && idx < maxindex
                idx += 1
            elseif r > 1-prob && idx > 1
                idx -= 1
            end
            indices[i] = idx
        end#if
        return nothing
    end

    rng = GPUArrays.default_rng(CuArray)

    kernel  = @cuda launch=false kernel(indices, maxindex, prob, rng.state)
    config  = launch_configuration(kernel.fun)
    threads = Base.min(length(indices), config.threads, 256)
    blocks  = cld(length(indices), threads)
    kernel(indices, maxindex, prob, rng.state; threads=threads, blocks=blocks)
    return indices
end
