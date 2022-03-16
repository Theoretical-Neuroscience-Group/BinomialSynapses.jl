"""
    jitter!(model::BinomialGridModel, width)

Apply jitter with parameter `width` to the indices of a binomial model on a grid.
"""
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

function jitter!(indices::AnyCuArray, maxindex, prob1, prob2)
    function kernel(indices, maxindex, prob1, prob2)
        # grid-stride loop
        tid    = threadIdx().x
        window = (blockDim().x - 1i32) * gridDim().x
        offset = (blockIdx().x - 1i32) * blockDim().x
        while offset < length(indices)
            i = tid + offset
            r = rand(Float32)
            if i <= length(indices)
                @inbounds idx = indices[i]
                if idx == 1
                    r < prob2 && (idx += 1)
                elseif idx == maxindex
                    r < prob2 && (idx -= 1)
                elseif r < prob1
                    idx += 1
                elseif r > 1-prob1
                    idx -= 1
                end
                @inbounds indices[i] = idx
            end#if
            offset += window
        end
        return nothing
    end

    kernel  = @cuda launch=false kernel(indices, maxindex, prob1, prob2)
    config  = launch_configuration(kernel.fun)
    threads = max(32, min(config.threads, length(indices)))
    blocks  = cld(length(indices), threads)
    kernel(indices, maxindex, prob1, prob2; threads=threads, blocks=blocks)
    return indices
end
