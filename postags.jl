include("util.jl")

using PyCall
nltk = PyCall.pywrap(PyCall.pyimport("nltk"))

using DataFrames

df = DataFrame(word=Any[], tag=Any[], count=Any[])
d = Dict()
count = Dict()

grid, jelly, l = getallinstructions()

for ins in vcat(grid, jelly, l)
    l = nltk.pos_tag(ins.text)
    for (w,t) in l
        d[w] = t
        if haskey(count, w)
            count[w] = count[w] + 1
        else
            count[w] = 1
        end
    end
end

for w in keys(d)
    push!(df, (w, d[w], count[w]))
end

writetable("data/postags.csv", df)
