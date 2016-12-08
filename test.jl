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

function testparagraph()
	grid, jelly, l = getallinstructions()

	p1s = Set()
	prev_p1 = ""
	prev_p2 = ""
	for ins in grid
		p1, p2 = split(ins.id, "-")
		if prev_p1 == ""
			prev_p1 = p1
			prev_p2 = p2
			push!(p1s, p1)
		else
			if p1 != prev_p1
				@test !(p1 in p1s)
				prev_p2 = p2
				prev_p1 = p1
				push!(p1s, p1)
			else
				parse(Int, p2) > parse(Int, prev_p2)
				prev_p2 = p2
			end
		end
	end
	
	println(p1s)
	
	p1s = Set()
	prev_p1 = ""
	prev_p2 = ""
	for ins in jelly
		p1, p2 = split(ins.id, "-")
		if prev_p1 == ""
			prev_p1 = p1
			prev_p2 = p2
			push!(p1s, p1)
		else
			if p1 != prev_p1
				@test !(p1 in p1s)
				prev_p2 = p2
				prev_p1 = p1
				push!(p1s, p1)
			else
				parse(Int, p2) > parse(Int, prev_p2)
				prev_p2 = p2
			end
		end
	end

	println(p1s)

	p1s = Set()
	prev_p1 = ""
	prev_p2 = ""
	for ins in l
		p1, p2 = split(ins.id, "-")
		if prev_p1 == ""
			prev_p1 = p1
			prev_p2 = p2
			push!(p1s, p1)
		else
			if p1 != prev_p1
				@test !(p1 in p1s)
				prev_p2 = p2
				prev_p1 = p1
				push!(p1s, p1)
			else
				parse(Int, p2) > parse(Int, prev_p2)
				prev_p2 = p2
			end
		end
	end

	println(p1s)

end

function testmerging()
	grid, jelly, l = getallinstructions()

	function check_consecutive(t1, t2)
		passed = true
		passed = passed | (abs(t1[1] - t2[1]) == 1 && t1[2] == t2[2] && t1[3] == t2[3])
		passed = passed | (abs(t1[2] - t2[2]) == 1 && t1[1] == t2[1] && t1[3] == t2[3])
		passed = passed | ((t1[3] + 90) % 360 == t2[3] && t1[1] == t2[1] && t1[2] == t2[2])
		passed = passed | ((t1[3] - 90) % 360 == t2[3] && t1[1] == t2[1] && t1[2] == t2[2])
		return passed
	end

	for s in [grid, jelly, l]
		merged = merge_singles(s)
		for ins in merged
			for i=2:length(ins.path)
				@test check_consecutive(ins.path[i-1], ins.path[i])
			end
		end
	end
end

#teststate()
#testmb()
#testget3()
#testparagraph()
testmerging()
