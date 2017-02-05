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
	emb[w] = getvec(model, w)
end

println("Saving...")
save("data/embeddings.jld", "vectors", emb)
