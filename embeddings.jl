include("wvec.jl")
include("util.jl")

using JLD, WordVec

println("Reading embeddings...")
@time model = Wvec("data/GoogleNews-vectors-negative300.bin")

println("Reading instructions...")
@time grid, jelly, l = getallinstructions()

println("Building vocab...")
@time vocab =  build_dict(vcat(grid, jelly, l))

emb = Dict()

for w in keys(vocab)
    t = w
    if contains(w, "rack")
        t = "rack"
    elseif w == "grey"
        t = "gray"
    elseif w in ["a", "of", "to", "and"]
        t = string("_", w)
    end

    vec, unk = getvec(model, t)
    emb[w] = vec
    if unk
        println(w)
    end
end

emb["unk"] = getvec(model, "unk")

println("Saving...")
save("data/embeddings.jld", "vectors", emb)
