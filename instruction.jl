using JLD

type Instruction
	fname #file name
	text #instruction as a list of tokens
	path #path as a list of (x,y,orientation) tuples
	map #map name
	id 
end

function getallinstructions(;fname="data/pickles/databag3.jld")
	raw_data = load(fname, "raw_data")

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

function readinsjson(fname)
    instructions = JSON.parsefile(fname)

    function json2ins(j)
        ind1 = findfirst(j["path"], '[')
        ind2 = findfirst(j["path"], ']')
        i = Instruction(j["fname"], j["text"], eval(parse(j["path"][ind1:ind2])),
                        j["map"], j["id"])
        return i
    end

    data = map(json2ins, instructions)

    return data
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
				push!(prg, Instruction(prev_ins.fname, text, path, prev_ins.map, prev_p1))
				prev_p2 = p2
				prev_p1 = p1
				text = Any[]
				path = Any[]
				append!(text, ins.text)
				append!(path, ins.path[1:end-1])
				prev_ins = ins
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

function group_singles(singles)
	prev_p1 = ""
	prev_p2 = ""
	prg = Any[]
	groups = Any[]

	for ins in singles
		p1, p2 = split(ins.id, "-")
		if prev_p1 == ""
			prev_p1 = p1
			prev_p2 = p2
			push!(prg, ins)
		else
			if p1 != prev_p1
				push!(groups, prg)
				prg = Any[]
				push!(prg, ins)
				prev_p2 = p2
				prev_p1 = p1
			else
				prev_p1 = p1
				prev_p2 = p2
				push!(prg, ins)
			end
		end
	end
	if length(prg) != 0; push!(groups, prg); end
	return groups
end
