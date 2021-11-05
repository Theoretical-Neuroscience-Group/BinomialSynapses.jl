struct Recording{T1, T2, T3}
    f1::T1
    f2::T2
    data::T3
end

struct NoRecording end

function update!(rec::Recording, sim, time)
    push!(rec.data, rec.f1(sim, time))
    return rec
end

update!(::NoRecording, _, _) = nothing

function save(rec::Recording)
    rec.f2(rec.data)
end

save(::NoRecording) = nothing
