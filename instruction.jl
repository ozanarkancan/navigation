using JLD

type Instruction
	fname #file name
	text #instruction as a list of tokens
	path #path as a list of (x,y,orientation) tuples
	map #map name
	id 
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
				ins["map"], ins["id"]))
		end
	end
	grid, jelly, l
end

function merge_singles(singles)
	prev_p1 = ""
	prev_p2 = ""
	text = Any[]
	path = Any[]
	prg = Any[]
	prev_ins = nothing

	for ins in singles
		p1, p2 = split(ins.id, "-")
		if prev_p1 == ""
			prev_p1 = p1
			prev_p2 = p2
			append!(text, ins.text)
			append!(path, ins.path[1:end-1])
			prev_ins = ins
		else
			if p1 != prev_p1
				append!(path, [prev_ins.path[end]])
				push!(prg, Instruction("", text, path, prev_ins.map, prev_p1))
				prev_p2 = p2
				prev_p1 = p1
				text = Any[]
				path = Any[]
			else
				prev_p1 = p1
				prev_p2 = p2
				append!(text, ins.text)
				append!(path, ins.path[1:end-1])
				prev_ins = ins
			end
		end
	end
	return prg
end
