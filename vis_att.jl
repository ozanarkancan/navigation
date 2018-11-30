using CSV, TSne, Random, Plots

function get_label(task, text, correct)
    checks = []
    push!(checks, occursin("stool", text))
    push!(checks, occursin("chair", text))
    push!(checks, occursin("easel", text))
    push!(checks, occursin("rack", text))
    push!(checks, occursin("lamp", text))
    push!(checks, occursin("sofa", text) || occursin("bench", text))
    push!(checks, occursin("octagon", text) || occursin("blue", text))
    push!(checks, occursin("brick", text))
    push!(checks, occursin("concrete", text) || occursin("cement", text))
    push!(checks, occursin("flower", text) || occursin("rose", text))
    push!(checks, occursin("grass", text))
    push!(checks, occursin("gravel", text))
    push!(checks, occursin("wood", text))
    push!(checks, occursin("yellow", text))
    push!(checks, occursin("butterflies", text))
    push!(checks, occursin("fish", text))
    push!(checks, occursin("tower", text))

    if sum(checks) == 1 && correct == 1.0
        return findmax(checks)[2]
    else
        return 7
    end
end

df = CSV.read("turntox.csv"; Dict=(:task=>String, :text=>String, :correct=>Float32, :firstatt=>String, :lastatt=>String));
df[:firstatt] = map(x->eval(Meta.parse(x)), df[:firstatt])
df[:lastatt] = map(x->eval(Meta.parse(x)), df[:lastatt])

Random.seed!(123456)
r = randperm(size(df, 1))

labels = map(x->get_label(df[:task][x], df[:text][x], df[:correct][x]), r);

cs = (:crimson, :cyan2)


colors = map(x->df[:task][x] == "move_to_x" ? cs[1] : cs[2], r)
indices = findall(x->x<7, labels)
l = min(2000, length(indices))
indices = indices[1:l]
labels=labels[indices]
#colors = colors[indices]
colors = Int.(labels)
colors = reshape(colors, 1, length(colors))

markers = [:star8, :pentagon, :circle, :star5, :rect, :cross]
labs = map(x->markers[x], labels)
labs = reshape(labs, 1, length(labs))

X = df[:firstatt][r[indices]]
X = convert(Matrix{Float32}, vcat(X...))

Y = tsne(X, 2, 0, 1500, 20.0)

p = scatter(reshape(Y[:,1], 1, l), reshape(Y[:,2], 1, l), m=labs, lab="", color=colors)
savefig(p,"firstatt_20_3.svg")

