using Base.Test
include("util.jl")

function teststate()
	fname = "data/maps/map-grid.json"
	grid = getmap(fname)

	fname = "data/maps/map-jelly.json"
	jelly = getmap(fname)
	
	fname = "data/maps/map-l.json"
	l = getmap(fname)
	
	fname = "data/instructions/ParagraphRandom.grid.json"
	instructions_grid = getinstructions(fname)
	
	fname = "data/instructions/ParagraphRandom.jelly.json"
	instructions_jelly = getinstructions(fname)
	
	fname = "data/instructions/ParagraphRandom.l.json"
	instructions_l = getinstructions(fname)

	cps = [(grid, instructions_grid), (jelly, instructions_jelly), (l, instructions_l)]
	for (map, inss) in cps
		println("Test $(map.name) Instructions")
		for ins in inss
			for loc in ins.path
				try
					view = state_agent_centric(map, loc; vdims = [39 39]);
				catch y
					println("Error: $(y)")
					println("fname: $(ins.fname)")
					println("text: $(ins.text)")
					println("path: $(ins.path)")
					println("map: $(ins.map)\n")
				end
			end
		end
	end

end

function testmb()
	trainfiles = ["data/instructions/SingleSentenceZeroInitial.grid.json","data/instructions/SingleSentenceZeroInitial.jelly.json"]
	fname = "data/maps/map-grid.json"
	grid = getmap(fname)

	fname = "data/maps/map-jelly.json"
	jelly = getmap(fname)

	fname = "data/maps/map-l.json"
	l = getmap(fname)

	maps = Dict("Grid" => grid, "Jelly" => jelly, "L" => l)

	trn_ins = getinstructions(trainfiles[1])
	append!(trn_ins, getinstructions(trainfiles[2]))

	println("Building the vocab...")
	vocab = build_dict(trn_ins)
	#vocab = build_char_dict(trn_ins)

	println("Converting data...")
	trn_data = map(x -> build_instance(x, maps[x.map], vocab), trn_ins)
	mb = minibatch(trn_data; bs=50)
	println("Length: $(length(mb))")
	println(size(mb[1][1]))
	for i=1:50
		println(indmax(mb[1][1][1][i, :]))
	end

	for k in keys(vocab)
		v = vocab[k]
		if v in [43, 15, 277]
			println("k,v: $((k,v))")
		end
	end
end

function testget3()
	grid, jelly, l = getallinstructions();

	for ins in grid
		@test ins.map == "Grid"
	end

	for ins in jelly
		@test ins.map == "Jelly"
	end

	for ins in l
		@test ins.map == "L"
	end


end

#teststate()
#testmb()
testget3()
