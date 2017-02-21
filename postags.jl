include("util.jl")

using PyCall
nltk = PyCall.pywrap(PyCall.pyimport("nltk"))

using DataFrames

df = DataFrame(word=Any[], tag=Any[])
d = Dict()

grid, jelly, l = getallinstructions()

for ins in vcat(grid, jelly, l)
    l = nltk.pos_tag(ins.text)
    for (w,t) in l
        d[w] = t
    end
end

for w in keys(d)
    push!(df, (w, d[w]))
end

writetable("data/postags.csv", df)
