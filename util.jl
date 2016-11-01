using JLD

include("instruction.jl")
include("map.jl")

function build_dict(instructions)
	d = Dict{AbstractString, Int}()

	for ins in instructions
		for w in ins.text
			for t in split(w, "-")
				get!(d, t, 1+length(d))
			end
		end
	end

	return d
end

function build_char_dict(instructions)
	d = Dict{Char, Int}()
	get!(d, ' ', 1+length(d))

	for ins in instructions
		for w in ins.text
			for t in split(w, "-")
				for c in t
					get!(d, c, 1+length(d))
				end
			end
		end
	end

	return d
end


#converts tokens to onehots
function ins_arr(d, ins)
	arr = Any[]
	vocablength = length(d)

	for w in ins
		for t in split(w, "-")
			indx = haskey(d, t) ? d[t] : vocablength + 1
			onehot = zeros(Float32, vocablength + 1, 1)
			onehot[indx, 1] = 1.0
			push!(arr, onehot)
		end
	end

	words = zeros(Float32, length(arr), vocablength+1)
	for i=1:length(arr); words[i, :] = arr[i]; end

	return words
end

#converts chars to onehots
function ins_char_arr(d, ins)
	arr = Any[]
	vocablength = length(d)

	for i=1:length(ins)
		t = i==length(ins) ? ins[i] : "$(ins[i]) "
		for c in t
			indx = haskey(d, c) ? d[c] : vocablength + 1
			onehot = zeros(Float32, vocablength + 1, 1)
			onehot[indx, 1] = 1.0
			push!(arr, onehot)
		end
	end
	return arr
end


#=
builds the agent's view
agent centric
up: front
right hand side: right of the agent
down: back
left hand side: left of the agent
(20, 20) is the agent curent location and it is a node
neighbors of a node are edges
=#
function state_agent_centric(map, loc; vdims = [39 39])
	lfeatvec = length(Items) + length(Floors) + length(Walls) + 1
	view = zeros(Float32, vdims[1], vdims[2], lfeatvec, 1)
	mid = [vdims[1] round(Int, vdims[2]/2)]
	
	if loc[3] == 0
		ux = 0; uy = -1;
		rx = 1; ry = 0;
		dx = 0; dy = 1;
		lx = -1; ly = 0;
	elseif loc[3] == 90
		ux = 1; uy = 0;
		rx = 0; ry = 1;
		dx = -1; dy = 0;
		lx = 0; ly = -1;
	elseif loc[3] == 180
		ux = 0; uy = 1;
		rx = -1; ry = 0;
		dx = 0; dy = -1;
		lx = 1; ly = 0;
	else
		ux = -1; uy = 0;
		rx = 0; ry = -1;
		dx = 1; dy = 0;
		lx = 0; ly = 1;
	end
	
	current = loc[1:2]

	i, j = mid
	
	view[i, j, map.nodes[(loc[1], loc[2])]] = 1.0

	i = i - 1
	#up
	while i > 0
		next = (current[1] + ux, current[2] + uy)
		if haskey(map.edges[(current[1], current[2])], (next[1], next[2]))#check the wall existence
			wall, floor = map.edges[(current[1], current[2])][(next[1], next[2])]
			view[i, j, length(Items) + floor, 1] = 1.0
			view[i, j, length(Items) + length(Floors) + wall, 1] = 1.0
			i = i - 1
			view[i, j, map.nodes[(next[1], next[2])], 1] = 1.0
			current = next
			i = i - 1
		else
			view[i, j, lfeatvec, 1] = 1.0
			break
		end
	end

	i, j = mid
	j = j + 1
	#right
	while j <= vdims[2]
		next = (current[1] + rx, current[2] + ry)
		if haskey(map.edges[(current[1], current[2])], (next[1], next[2]))#check the wall existence
			wall, floor = map.edges[(current[1], current[2])][(next[1], next[2])]
			view[i, j, length(Items) + floor, 1] = 1.0
			view[i, j, length(Items) + length(Floors) + wall, 1] = 1.0
			j = j + 1
			view[i, j, map.nodes[(next[1], next[2])], 1] = 1.0
			current = next
			j = j + 1
		else
			view[i, j, lfeatvec, 1] = 1.0
			break
		end
	end

	i, j = mid
	i = i + 1
	#down
	while i <= vdims[1]
		next = (current[1] + dx, current[2] + dy)
		if haskey(map.edges[(current[1], current[2])], (next[1], next[2]))#check the wall existence
			wall, floor = map.edges[(current[1], current[2])][(next[1], next[2])]
			view[i, j, length(Items) + floor] = 1.0
			view[i, j, length(Items) + length(Floors) + wall] = 1.0
			i = i + 1
			view[i, j, map.nodes[(next[1], next[2])]] = 1.0
			current = next
			i = i + 1
		else
			view[i, j, lfeatvec] = 1.0
			break
		end
	end

	i, j = mid
	j = j - 1
	#left
	while j > 0
		next = (current[1] + lx, current[2] + ly)
		if haskey(map.edges[(current[1], current[2])], (next[1], next[2]))#check the wall existence
			wall, floor = map.edges[(current[1], current[2])][(next[1], next[2])]
			view[i, j, length(Items) + floor] = 1.0
			view[i, j, length(Items) + length(Floors) + wall] = 1.0
			j = j - 1
			view[i, j, map.nodes[(next[1], next[2])]] = 1.0
			current = next
			j = j - 1
		else
			view[i, j, lfeatvec] = 1.0
			break
		end
	end

	return view
end

function action(curr, next)
	ygold = zeros(Float32, 4, 1)
	
	if curr[1] != next[1] || curr[2] != next[2]#move
		ygold[1] = 1.0
	elseif next[3] > curr[3] || (next[3] == 0 && curr[3] == 270)#right
		ygold[2] = 1.0
	elseif next[3] < curr[3] || (next[3] == 270 && curr[3] == 0)#left
		ygold[3] = 1.0
	else
		ygold[4] = 1.0
	end

	return ygold
end

function build_instance(instance, map, vocab; vdims=[39, 39])
	words = ins_arr(vocab, instance.text)
	#ins = ins_char_arr(vocab, instance.text)

	lfeatvec = length(Items) + length(Floors) + length(Walls) + 1
	states = Any[]
	Y = zeros(Float32, length(instance.path), 4)

	for i=1:length(instance.path)
		curr = instance.path[i]
		next = i == length(instance.path) ? curr : instance.path[i+1]
		Y[i, :] = action(curr, next)
		push!(states, state_agent_centric(map, curr))
	end

	return (words, states, Y)
end

function build_data(trainfiles, outfile)
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
	
	println("Saving...")

	save(outfile, "vocab", vocab, "maps", maps, "data", trn_data)
	println("Done!")
end
