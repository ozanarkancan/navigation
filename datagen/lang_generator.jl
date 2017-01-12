opposites = Dict(0=>"south", 90=>"west", 180=>"north", 270=>"east")
rights = Dict(0=>"east", 90=>"south", 180=>"west", 270=>"north")
lefts = Dict(0=>"west", 90=>"north", 180=>"east", 270=>"south")
ordinals = Dict(1=>"first", 2=>"second", 3=>"third", 4=>"fourth", 5=>"fifth", 
	6=>"sixth", 7=>"seventh", 8=>"eighth", 9=>"ninth")
times = Dict(1=>"once", 2=>"twice")
numbers = Dict(1=>["one", "a"],2=>["two"],3=>["three"],4=>["four"],5=>["five"],
	6=>["six"],7=>["seven"],8=>["eight"],9=>["nine"],10=>["ten"])

function action(curr, next)
	a = 0
	if curr[1] != next[1] || curr[2] != next[2]#move
		a = 1
	elseif next[3] > curr[3] || (next[3] == 0 && curr[3] == 270)#right
		a = 2
	elseif next[3] < curr[3] || (next[3] == 270 && curr[3] == 0)#left
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
	while ind < length(segments)-1
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
		txt = string(txt, "\n", s, "\n", ins, "\n\n")
	end
	return txt
end

function startins(navimap, maze, curr, next)
	curr_t, curr_s = curr
	next_t, next_s = next

	a = action(curr_s[1], curr_s[2])

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
			dir = opposites[curr_s[1][3]]
		end
		push!(cands, string("turn to the ", dir))
		push!(cands, string("orient yourself to the ", dir))
		push!(cands, string("turn to face the ", dir))
		return  [(curr_s, rand(cands))]
	else
		return moveins(navimap, maze, curr, next)
	end
end

function moveins(navimap, maze, curr, next)
	curr_t, curr_s = curr
	next_t, next_s = next != nothing ? next : nothing, nothing
	endpoint = map(x->round(Int, x), curr_s[end])
	d = round(Int, endpoint[3] / 90 + 1)

	cands = Any[]
	steps = length(curr_s)-1
	push!(cands, string(rand(["go ", "move "]), rand(["forward ", "straight "]),
		rand(numbers[steps]), (steps > 1 ? rand([" steps", " blocks", " segments"]) : rand([" step", " block", " segment"]))))
	push!(cands, string("take ", rand(numbers[steps]),
		(steps > 1 ? rand([" steps", " blocks", " segments"]) : rand([" step", " block", " segment"]))))

	faceWall = maze[endpoint[2], endpoint[1], d] == 0
	if faceWall
		push!(cands, string(rand(["move ", "go "]), rand(["forward ", "straight ", ""]),
			"until the wall"))
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
	return [(curr_s, rand(cands))]
end

function finalins(navimap, maze, curr)
	curr_t, curr_s = curr
	cands = Any[]

	lasti = ""

	if curr_t == "turn"
		insl = turnins(navimap, maze, curr, nothing)
	else
		insl = moveins(navimap, maze, curr, nothing)
	end
	lasts, lasti = insl[end]

	if rand() < 0.5
		push!(cands, string(lasti, " and that is the ", rand(["target ", "final "]), "position"))
		push!(cands, string(lasti, " and that is the position ", rand(numbers[rand(2:10)])))
		push!(cands, string(lasti, " and there should be the position ", rand(numbers[rand(2:10)])))
		insl[end] = (lasts, rand(cands))
		return insl
	else
		push!(cands, string("that is the ", rand(["target ", "final "]), "position"))
		push!(cands, string("that is the position ", rand(numbers[rand(2:10)])))
		push!(cands, string("there should be the position ", rand(numbers[rand(2:10)])))

		push!(insl, ([curr_s[end]], rand(cands)))
		return insl
	end
end
