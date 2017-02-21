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
        if haskey(d, w)
            if haskey(d[w], t)
                d[w][t] = d[w][t] + 1
            else
                d[w][t] = 1
            end
        else
            d[w] = Dict()
            d[w][t] = 1
        end

        if haskey(count, w)
            count[w] = count[w] + 1
        else
            count[w] = 1
        end
    end
end

for w in keys(d)
    println("Word: $w -> $(d[w])")
    t = sort(collect(d[w]);rev=true, by=x->x[2])[1][1]
    push!(df, (w, t, count[w]))
end

writetable("data/postags.csv", df)
