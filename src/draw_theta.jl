using CUDA

function kernel_draw_theta!(indexes,N_range,p_range,q_range,sigma_range,tau_range,jittering_kernel_width,r,Ns,ps,qs,sigmas,taus)

    M_out, n_parameters = size(indexes)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i <= M_out

        # Initializes the indexes
        index_N = indexes[i,1]
        index_p = indexes[i,2]
        index_q = indexes[i,3]
        index_sigma = indexes[i,4]
        index_tau = indexes[i,5]

        # Initializes the random flips
        random_flip_N = 0
        random_flip_p = 0
        random_flip_q = 0
        random_flip_sigma = 0
        random_flip_tau = 0
        random_flip_N_edge = 0
        random_flip_p_edge = 0
        random_flip_q_edge = 0
        random_flip_sigma_edge = 0
        random_flip_tau_edge = 0

        # Draws the random flips
        if r[i,1] < 1/(jittering_kernel_width-1)
            random_flip_N_edge = 1
        end
        if r[i,2] < 1/(jittering_kernel_width-1)
            random_flip_p_edge = 1
        end
        if r[i,3] < 1/(jittering_kernel_width-1)
            random_flip_q_edge = 1
        end
        if r[i,4] < 1/(jittering_kernel_width-1)
            random_flip_sigma_edge = 1
        end
        if r[i,5] < 1/(jittering_kernel_width-1)
            random_flip_tau_edge = 1
        end

        if r[i,1] < 1/jittering_kernel_width
            random_flip_N = 1
        elseif r[i,1] > 1-1/jittering_kernel_width
            random_flip_N = -1
        end

        if r[i,2] < 1/jittering_kernel_width
            random_flip_p = 1
        elseif r[i,2] > 1-1/jittering_kernel_width
            random_flip_p = -1
        end

        if r[i,3] < 1/jittering_kernel_width
            random_flip_q = 1
        elseif r[i,3] > 1-1/jittering_kernel_width
            random_flip_q = -1
        end

        if r[i,4] < 1/jittering_kernel_width
            random_flip_sigma = 1
        elseif r[i,4] > 1-1/jittering_kernel_width
            random_flip_sigma = -1
        end

        if r[i,5] < 1/jittering_kernel_width
            random_flip_tau = 1
        elseif r[i,5] > 1-1/jittering_kernel_width
            random_flip_tau = -1
        end

        # Computes the new indexes
        if indexes[i,1] == 1
            index_N = indexes[i,1]+random_flip_N_edge
        elseif indexes[i,1] == length(N_range)
            index_N = indexes[i,1]-random_flip_N_edge
        else
            index_N = indexes[i,1]+random_flip_N
        end
        if indexes[i,2] == 1
            index_p = indexes[i,2]+random_flip_p_edge
        elseif indexes[i,2] == length(p_range)
            index_p = indexes[i,2]-random_flip_p_edge
        else
            index_p = indexes[i,2]+random_flip_p
        end
        if indexes[i,3] == 1
            index_q = indexes[i,3]+random_flip_q_edge
        elseif indexes[i,3] == length(q_range)
            index_q = indexes[i,3]-random_flip_q_edge
        else
            index_q = indexes[i,3]+random_flip_q
        end
        if indexes[i,4] == 1
            index_sigma = indexes[i,4]+random_flip_sigma_edge
        elseif indexes[i,4] == length(sigma_range)
            index_sigma = indexes[i,4]-random_flip_sigma_edge
        else
            index_sigma = indexes[i,4]+random_flip_sigma
        end
        if indexes[i,5] == 1
            index_tau = indexes[i,5]+random_flip_tau_edge
        elseif indexes[i,5] == length(tau_range)
            index_tau = indexes[i,5]-random_flip_tau_edge
        else
            index_tau = indexes[i,5]+random_flip_tau
        end

        indexes[i,1] = index_N
        indexes[i,2] = index_p
        indexes[i,3] = index_q
        indexes[i,4] = index_sigma
        indexes[i,5] = index_tau

        # Returns the values from the indexes
        Ns[i] = N_range[index_N]
        ps[i] = p_range[index_p]
        qs[i] = q_range[index_q]
        sigmas[i] = sigma_range[index_sigma]
        taus[i] = tau_range[index_tau]
    end
    return nothing
end

function draw_theta!(indexes,N_range,p_range,q_range,sigma_range,tau_range,jittering_kernel_width,Ns,ps,qs,sigmas,taus)
    M_out, n_parameters = size(indexes)
    r = CUDA.rand(M_out, n_parameters)
    kernel  = @cuda name="draw_theta" launch=false kernel_draw_theta!(indexes,N_range,p_range,q_range,sigma_range,tau_range,jittering_kernel_width,r,Ns,ps,qs,sigmas,taus)
    config  = launch_configuration(kernel.fun)
    threads = Base.min(M_out, config.threads, 256)
    blocks  = cld(M_out, threads)
    kernel(indexes,N_range,p_range,q_range,sigma_range,tau_range,jittering_kernel_width,r,Ns,ps,qs,sigmas,taus; threads=threads, blocks=blocks)
end
