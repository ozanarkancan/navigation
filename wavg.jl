using StatsBase

s = [874, 1293, 1070] ./ 3237
p = [224, 242, 236] ./ 702

s = WeightVec(s)
p = WeightVec(p)

singles = [parse(Float32, ARGS[1]), parse(Float32, ARGS[2]), parse(Float32, ARGS[3])]
paragraphs = [parse(Float32, ARGS[4]), parse(Float32, ARGS[5]), parse(Float32, ARGS[6])]

println("Single: $(mean_and_std(singles, s))")
println("Paragraph: $(mean_and_std(paragraphs, p))")
