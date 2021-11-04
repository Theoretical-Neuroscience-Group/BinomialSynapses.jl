struct Recording{T1, T2, T3}
    f1::T1
    f2::T2
    data::T3
end

function Recording(f1, f2, sim::NestedFilterSimulation)
    res = f1(sim)
    data = [res]
    return Recording(f1, f2, data)
end

function update!(rec::Recording, sim, i, time)
    push!(rec.data, rec.f1(sim, time, i))
    return rec
end

function save(rec::Recording)
    rec.f2(rec.data)
end
