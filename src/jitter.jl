using CUDA

function jitter!(model::BinomialGridModel, width)
    prob1 = Float32(1/width)
    prob2 = Float32(1/(width-1))

    jitter!(model.Nind, length(model.Nrng), prob1, prob2)
    jitter!(model.pind, length(model.prng), prob1, prob2)
    jitter!(model.qind, length(model.qrng), prob1, prob2)
    jitter!(model.σind, length(model.σrng), prob1, prob2)
    jitter!(model.τind, length(model.τrng), prob1, prob2)

    refresh!(model)
    return model
end

function jitter!(indices, maxindex, prob1, prob2)
    function kernel(indices, range, prob1, prob2, randstates)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        @inbounds if i <= length(indices)
            r   = GPUArrays.gpu_rand(Float32, CUDA.CuKernelContext(), randstates)
            idx = indices[i]
            if idx == 1
                r < prob2 && (idx += 1)
            elseif idx == maxindex
                r < prob2 && (idx -= 1)
            elseif r < prob1
                idx += 1
            elseif r > 1-prob1
                idx -= 1
            end
            indices[i] = idx
        end#if
        return nothing
    end

    rng = GPUArrays.default_rng(CuArray)

    kernel  = @cuda launch=false kernel(indices, maxindex, prob1, prob2, rng.state)
    config  = launch_configuration(kernel.fun)
    threads = Base.min(length(indices), config.threads, 256)
    blocks  = cld(length(indices), threads)
    kernel(indices, maxindex, prob1, prob2, rng.state; threads=threads, blocks=blocks)
    return indices
end
