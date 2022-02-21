"""
    Recording(f1, f2, data)

A recording, which is stored in `data`. The function `f1` is something that is computed at each time step, whereas `f2` is an operation that is applied after the simulation is finished.
"""
struct Recording{T1, T2, T3}
    f1::T1
    f2::T2
    data::T3
end

"""
    update!(rec::Recording, sim, time)

Compute `f1(sim, time)` and store it in `rec.data`.
"""
function update!(rec::Recording, sim, time)
    update_data!(rec.data, rec.f1, sim, time)
    return rec
end

update_data!(data, f1, sim, time) = push!(data, f1(sim, time))
update_data!(_, ::Nothing, _, _) = nothing

"""
    save(rec::Recording)

Apply `f2` to `data`.
"""
function save(rec::Recording)
    save_data(rec.data, rec.f2)
end

save_data(data, f2) = f2(data)
save_data(::Nothing, _) = nothing
save_data(_, ::Nothing) = nothing
save_data(::Nothing, ::Nothing) = nothing

"""
    NoRecording

Do not record any data about the simulation (except what is stored in the simulation object already).
"""
const NoRecording = Recording(nothing, nothing, nothing)
