include("util.jl")

files = ["data/instructions/SingleSentenceZeroInitial.grid.json",
"data/instructions/SingleSentenceZeroInitial.jelly.json"]
build_data(files, "grid_jelly.jld")

files = ["data/instructions/SingleSentenceZeroInitial.grid.json",
"data/instructions/SingleSentenceZeroInitial.l.json"]
build_data(files, "grid_l.jld")

files = ["data/instructions/SingleSentenceZeroInitial.l.json",
"data/instructions/SingleSentenceZeroInitial.jelly.json"]
build_data(files, "l_jelly.jld")
