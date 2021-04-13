function likelihood_resample!(state::BinomialState, model, observation)
    u, idx = likelihood_indices(state.k, model, observation)
    state.n .= state.n[idx]
    state.k .= state.k[idx]
    return u
end
