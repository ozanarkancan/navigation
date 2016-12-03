using JSON, TextAnalysis
using DataStructures
using JLD

type Instruction
	fname #file name
	text #instruction as a list of tokens
	path #path as a list of (x,y,orientation) tuples
	map #map name
end

function getallinstructions(;fname="data/pickles/databag3.pickle")
	raw_data = load("data/pickles/databag3.jld", "raw_data")

	grid = Instruction[]
	jelly = Instruction[]
	l = Instruction[]

	for (mp, arr) in [("grid", grid), ("jelly", jelly), ("l", l)]
		instructions = raw_data[mp]
		for ins in instructions
			push!(arr, Instruction(ins["filename"],
				ins["instruction"],
				map(x->(x[1], x[2], x[3]), ins["cleanpath"]),
				ins["map"]))
		end
	end
	grid, jelly, l
end

function getinstructions(fname)
	j = JSON.parsefile(fname; dicttype=DataStructures.OrderedDict, use_mmap=true)
	examples = j["examples"]["example"]
	instructions = Instruction[]

	for example in examples
		sd = StringDocument(example["instruction"]["__text"])
		prepare!(sd, strip_punctuation)
		prepare!(sd, strip_whitespace)
		prepare!(sd, strip_case)

		pathtext = (isa(example["path"], DataStructures.OrderedDict{AbstractString,Any}) || isa(example["path"], DataStructures.OrderedDict{String,Any})) ? example["path"]["__text"] : example["path"]
		sd2 = StringDocument(pathtext)
		prepare!(sd2, strip_whitespace)
		path = eval(parse(text(sd2)))

		push!(instructions,
			Instruction(example["instruction"]["_filename"],
				tokens(sd), path, example["_map"]
				)
		)
	end

	return instructions
end

function main()
	fname = "data/instructions/ParagraphRandom.grid.json"
	instructions = getinstructions(fname)

	for ins in instructions
		println("fname: $(ins.fname)")
		println("text: $(ins.text)")
		println("path: $(ins.path)")
		println("map: $(ins.map)\n")
	end
end

#main()
