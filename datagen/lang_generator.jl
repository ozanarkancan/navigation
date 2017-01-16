#=
Convert rand choices to different candidates
=#
include("navimap_utils.jl")

opposites = Dict(0=>"south", 90=>"west", 180=>"north", 270=>"east")
rights = Dict(0=>"east", 90=>"south", 180=>"west", 270=>"north")
lefts = Dict(0=>"west", 90=>"north", 180=>"east", 270=>"south")
ordinals = Dict(1=>"first", 2=>"second", 3=>"third", 4=>"fourth", 5=>"fifth", 
	6=>"sixth", 7=>"seventh", 8=>"eighth", 9=>"ninth")
times = Dict(1=>"once", 2=>"twice")
numbers = Dict(1=>["one", "a"],2=>["two"],3=>["three"],4=>["four"],5=>["five"],
	6=>["six"],7=>["seven"],8=>["eight"],9=>["nine"],10=>["ten"])
wall_names = Dict(1=>"butterflies",2=>"fish",3=>"towers")
floor_names = Dict(1=>["octagon", "blue-tiled"],2=>["brick"],3=>["concrete"],4=>["flower"],
	5=>["grass"],6=>["gravel", "stone"],7=>["wood", "wooden"],8=>["yellow"])
item_names = Dict(1=>"barstool", 2=>"chair", 3=>"easel", 4=>"hatrack",
	5=>"lamp", 6=>"sofa")

function action(curr, next)
	a = 0
	if curr[1] != next[1] || curr[2] != next[2]#move
		a = 1
	elseif !(next[3] == 270 && curr[3] == 0) && (next[3] > curr[3] || (next[3] == 0 && curr[3] == 270))#right
		a = 2
	elseif !(next[3] == 0 && curr[3] == 270) && (next[3] < curr[3] || (next[3] == 270 && curr[3] == 0))#left
		a = 3
	else
		a = 4
	end
	return a
end				

function generate_lang(navimap, maze, segments)
	generation = Any[]

	if length(segments) > 1
		append!(generation, startins(navimap, maze, segments[1], segments[2]))
	end

	ind = 2
	while ind < length(segments)
		g = (segments[ind][1] == "turn" ? turnins : moveins)(navimap, maze, segments[ind], segments[ind+1])
		append!(generation, g)
		ind += 1
	end
	append!(generation, finalins(navimap, maze, segments[end]))
	return generation
end

function to_string(generation)
	txt = ""
	for (s, ins) in generation
		txt = string(txt, "\n", s, "\n", ins, "\n")
	end
	return txt
end

function startins(navimap, maze, curr, next)
	"""
	TODO
	["to", "towards"]
	"""
	curr_t, curr_s = curr
	next_t, next_s = next

	a = action(curr_s[1], curr_s[2])
	p1 = (curr_s[1][2], curr_s[1][1], -1)
	
	if curr_t == "turn"
		cands = Any[]
		dir = ""
		d = ""
		if length(curr_s) == 2
			if a == 2#right
				dir = rights[curr_s[1][3]]
				d = "right"
			else#left
				dir = lefts[curr_s[1][3]]
				d = "left"
			end
			push!(cands, string("turn ", d))
		else
			push!(cands, "turn around")
			dir = opposites[curr_s[1][3]]
		end
		push!(cands, string("turn to the ", dir))
		push!(cands, string("orient yourself to the ", dir))
		push!(cands, string("turn to face the ", dir))

		diff_w, diff_f = around_different_walls_floor(navimap, (curr_s[1][1], curr_s[1][2]))
		wpatrn, fpatrn = navimap.edges[(next_s[1][1], next_s[1][2])][(next_s[2][1], next_s[2][2])]
		
		if diff_w
			 push!(cands, string("face the ", rand(["corridor ", "hall ", "alley "]), 
			 	"with the ", wall_names[wpatrn], rand(["", " on the wall"])))
			 push!(cands, string("turn your face to the ", rand(["corridor ", "hall ", "alley "]),
			 	" with the ", wall_names[wpatrn], rand(["", " on the wall"])))
			 push!(cands, string("turn to the ", rand(["corridor ", "hall ", "alley "]),
			 	" with the ", wall_names[wpatrn], rand(["", " on the wall"])))
		end

		if diff_f
			push!(cands, string("face the ", rand([rand(floor_names[fpatrn]), rand(ColorMapping[fpatrn])]),
				rand([" path", " hall", " hallway", " alley", " corridor"])))
			push!(cands, string("turn your face to the ", rand([rand(floor_names[fpatrn]), rand(ColorMapping[fpatrn])]),
				rand([" path", " hall", " hallway", " alley", " corridor"])))
			push!(cands, string("turn until you see the ", rand([rand(floor_names[fpatrn]), rand(ColorMapping[fpatrn])]), 
				rand([" path", " hall", " hallway", " alley", " corridor"])))
			push!(cands, string("turn to the ", rand([rand(floor_names[fpatrn]), rand(ColorMapping[fpatrn])]), 
				rand([" path", " hall", " hallway", " alley", " corridor"])))
			push!(cands, string("you should be ", rand(["facing the ", "seeing the "]),
				rand([rand(floor_names[fpatrn]), rand(ColorMapping[fpatrn])]), 
				rand([" path", " hall", " hallway", " alley", " corridor"])))
		end

		if is_deadend(maze, p1)
			push!(cands, "you should leave the dead end")
			push!(cands, string("only one ", 
				rand(["way ", "direction "]), 
				"to ", 
				rand(["go", "move", "travel"])))
		end

		if sum(maze[p1[1], p1[2], :]) == 3 || sum(maze[p1[1], p1[2], :]) == 2
			p = (curr_s[end][2], curr_s[end][1], round(Int, 1+curr_s[end][3] / 90))
			rightwall = maze[p[1], p[2], rightof(p[3])] == 0
			leftwall = maze[p[1], p[2], leftof(p[3])] == 0
			backwall = maze[p[1], p[2], backof(p[3])] == 0
			
			if rightwall && !backwall && !leftwall
				push!(cands, "turn so that the wall is on your right")
			elseif rightwall && backwall && !leftwall
				push!(cands, "turn so that the wall is on your right and back")
				push!(cands, "turn so that the wall is on your back and right")
			elseif !rightwall && !backwall && leftwall
				push!(cands, "turn so that the wall is on your left")
			elseif !rightwall && backwall && leftwall
				push!(cands, "turn so that the wall is on your left and back")
				push!(cands, "turn so that the wall is on your back and left")
			elseif !rightwall && backwall && !leftwall
				push!(cands, "turn so that your back is to the wall")
				push!(cands, "turn so that your back faces the wall")
				push!(cands, string("place your back", rand([" to", " against"]), " the wall"))
			end
		end

		return  [(curr_s, rand(cands))]
	else
		diff_w, diff_f = around_different_walls_floor(navimap, (curr_s[1][1], curr_s[1][2]))
		wpatrn, fpatrn = navimap.edges[(curr_s[1][1], curr_s[1][2])][(curr_s[2][1], curr_s[2][2])]
		l = Any[]
		
		if diff_f && is_corner(maze, p1)
			push!(l, ([curr_s[1]], string("you should be ", rand(["facing the ", "seeing the "]),
				rand([rand(floor_names[fpatrn]), rand(ColorMapping[fpatrn])]), 
				rand([" path", " hall", " hallway", " alley", " corridor"]))))
		end
		
		append!(l, moveins(navimap, maze, curr, next))
		return l
	end
end

function moveins(navimap, maze, curr, next)
	curr_t, curr_s = curr
	next_t, next_s = next != nothing ? next : nothing, nothing
	endpoint = map(x->round(Int, x), curr_s[end])
	d = round(Int, endpoint[3] / 90 + 1)

	cands = Any[]
	steps = length(curr_s)-1

	push!(cands, string(rand(["go ", "move ", "walk "]), rand(["forward ", "straight ", " "]),
		rand(numbers[steps]), (steps > 1 ? rand([" steps", " blocks", " segments", " times"]) : rand([" step", " block", " segment"]))))
	push!(cands, string("take ", rand(numbers[steps]),
		(steps > 1 ? rand([" steps", " blocks", " segments"]) : rand([" step", " block", " segment"]))))
	
	push!(cands, string(rand(numbers[steps]),
		(steps > 1 ? rand([" steps", " blocks", " segments"]) : rand([" step", " block", " segment", " space"])), " forward"))

	if facing_wall(maze, (endpoint[2], endpoint[1], d))
		push!(cands, string(rand(["move ", "go ", "walk "]), rand(["forward ", "straight ", ""]),
			"until the wall"))
	end

	p1 = (curr_s[1][2], curr_s[1][1], -1)
	p2 = (curr_s[end][2], curr_s[end][1], -1)
	
	if is_corner(maze, p2)
		push!(cands, string(rand(["move", "go", "walk"]),
			rand(["", " forward", " straight"]), " into the corner"))
	end

	if is_deadend(maze, p2)
		push!(cands, string(rand(["move", "go", "walk"]),
			rand(["", " forward", " straight"]), " into the dead end"))
	end


	if (is_corner(maze, p1) || is_deadend(maze, p1)) && (is_corner(maze, p2) || is_deadend(maze, p2))
		push!(cands, string(rand(["move", "go", "walk"]), " to the other end", 
			rand(["", string(" of the ", rand(["hall", "hallway", "path", "corridor", "alley"]))])))
	elseif (is_corner(maze, p2) || is_deadend(maze, p2))
		push!(cands, string(rand(["move", "go", "walk"]), " to the end", 
			rand(["", string(" of the ", rand(["hall", "hallway", "path", "corridor", "alley"]))])))
	end

	if is_intersection(maze, p2)
		alleycnt = count_alleys(maze, curr_s)
		if alleycnt > 0
			push!(cands, string(rand(["move", "go", "walk"]), rand([" until the ", " to the "]), ordinals[alleycnt], " alley"))
		end
	end

	if navimap.nodes[curr_s[end][1:2]] != 7 && item_single_on_this_segment(navimap, curr_s)
		push!(cands, string(rand(["move ", "go ", "walk "]), rand(["forward ", "straight ", ""]),
			"until the ", item_names[navimap.nodes[curr_s[end][1:2]]]))
		push!(cands, string(rand(["move ", "go ", "walk "]), "towards the ", item_names[navimap.nodes[curr_s[end][1:2]]]))
		push!(cands, string("take the ", rand(["path", "hall"])," towards the ", item_names[navimap.nodes[curr_s[end][1:2]]]))

		wpatrn, fpatrn = navimap.edges[(curr_s[1][1], curr_s[1][2])][(curr_s[2][1], curr_s[2][2])]
		push!(cands, string(rand(["follow the ", "along the "]),
			rand([rand(floor_names[fpatrn]), rand(ColorMapping[fpatrn])]),
			rand([" path", " hall", " hallway", " alley", " corridor"]), " to the", 
			item_names[navimap.nodes[curr_s[end][1:2]]]))
	elseif navimap.nodes[curr_s[end][1:2]] == 7 && navimap.nodes[curr_s[end-1][1:2]] != 7 && 
			length(curr_s) > 2 && item_single_on_this_segment(navimap, curr_s[1:end-1])
		push!(cands, string(rand(["move ", "go "]), rand(["a", "one"]), rand([" step", " block", " segment"]),
			" beyond the ", item_names[navimap.nodes[curr_s[end-1][1:2]]]))
	end

	return [(curr_s, rand(cands))]
end

function turnins(navimap, maze, curr, next)
	curr_t, curr_s = curr
	cands = Any[]
	a = action(curr_s[1], curr_s[2])
	d = a == 2 ? "right" : "left"
	push!(cands, string("turn ", d))
	push!(cands, string("turn to the ", d))
	push!(cands, string("make a ", d))

	if is_corner(maze, (curr_s[1][2], curr_s[1][1], round(Int, curr_s[1][3]/90 + 1)))
		push!(cands, string("at the corner turn ", d))
	end

	return [(curr_s, rand(cands))]
end

function finalins(navimap, maze, curr)
	"""
	TODO
	dead end
	"""
	curr_t, curr_s = curr
	cands = Any[]

	lasti = ""

	if curr_t == "turn"
		insl = turnins(navimap, maze, curr, nothing)
	else
		insl = moveins(navimap, maze, curr, nothing)
	end
	lasts, lasti = insl[end]

	r = rand()
	if r < 0.2
		return insl
	elseif r <= 0.6
		num = rand([rand(numbers[rand(2:10)]), rand(2:10)])
		push!(cands, string(lasti, " and that is the ", rand(["target ", "final "]), "position"))
		push!(cands, string(lasti, " and that is the position ", num))
		push!(cands, string(lasti, " and there should be the position ", num))
		insl[end] = (lasts, rand(cands))
		return insl
	else
		num = rand([rand(numbers[rand(2:10)]), rand(2:10)])
		push!(cands, string("that is the ", rand(["target ", "final "]), "position"))
		push!(cands, string("that is the position ", num))
		push!(cands, string("there should be the position ", num))
		push!(cands, string("position ", num, " should be there"))

		push!(insl, ([curr_s[end]], rand(cands)))
		return insl
	end
end
