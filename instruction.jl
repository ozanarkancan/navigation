using JSON, TextAnalysis
using DataStructures

type Instruction
	fname #file name
	text #instruction as a list of tokens
	path #path as a list of (x,y,orientation) tuples
	map #map name
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

		pathtext = (isa(example["path"], DataStructures.OrderedDict{AbstractString,Any}) || isa(example["path"], DataStructures.OrderedDict{UTF8String,Any})) ? example["path"]["__text"] : example["path"]
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
