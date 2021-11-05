struct Recording{T1, T2, T3}
    f1::T1
    f2::T2
    data::T3
end

function update!(rec::Recording, sim, time)
    update_data!(rec.data, rec.f1, sim, time)
    return rec
end

update_data!(data, f1, sim, time) = push!(data, f1(sim, time))
update_data!(::Nothing, _, _, _) = nothing
update_data!(_, ::Nothing, _, _) = nothing
update_data!(::Nothing, ::Nothing, _, _) = nothing

function save(rec::Recording)
    save_data(rec.data, rec.f2)
end

save_data(data, f2) = f2(data)
save_data(::Nothing, _) = nothing
save_data(_, ::Nothing) = nothing
save_data(::Nothing, ::Nothing) = nothing

const NoRecording = Recording(nothing, nothing, nothing)
