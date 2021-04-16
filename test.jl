using BenchmarkTools, CUDA

m_out  = 1024
m_in   = 1024
Ns     = CuArray(rand(1:128, m_out))
ps     = CUDA.rand(m_out)
qs     = CUDA.rand(m_out)
sigmas = CUDA.rand(m_out)
taus   = CUDA.rand(m_out)
model  = BinomialModel(Ns, ps, qs, sigmas, taus);

ns = CuArray(rand(1:128, m_out, m_in))
ks = CUDA.zeros(Int, m_out, m_in);

ks .= ns .% 64

#propagate!(ns, ks, model, 0.1f0)

ks

CUDA.@time u1 = likelihood(ks, model, 0.3f0)[:,1]

CUDA.@time u2, idx = likelihood_indices(ks, model, 0.3f0)
CUDA.@time u2, idx2 = likelihood_indices(ks, model, 0.3f0)
u1 ≈ u2

idx == idx2


maximum(idx)
minimum(idx)

@benchmark CUDA.@sync likelihood($ks, $model, 0.3f0)[:,1]
@benchmark CUDA.@sync likelihood_indices($ks, $model, 0.3f0)

@device_code_warntype likelihood_indices(ks, model, 0.3f0)
