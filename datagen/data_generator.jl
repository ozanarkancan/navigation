include("../util.jl")
include("maze.jl")

function generate_path(maze)
	h,w,_ = size(maze)
	rp = randperm(h*w)

	x1,y1 = ind2sub((h,w), rp[1])
	z1 = rand(0:3) * 90
	x2 = 0
	y2 = 0
	dist = 0
	i=2

	while dist < 6
		x2,y2 = ind2sub((h,w), rp[i])
		i += 1
		dist = abs(x1-x2)+abs(y1-y2)
	end

	println("Start: $((y1, x1, z1))")
	println("Goal: $((y2, x2))")
	
	path = astar_solver(maze, [x1, y1], [x2, y2])

	start = (y1, x1, z1)
	goal = (y2, x2, -1)

	nodes = Any[]

	current = start
	next = 2
	#println("Path: $path")

	while !(current[1] == goal[1] && current[2] == goal[2])
		#println("Current: $current")
		y2, x2 = path[next]
		nb = (x2, y2)
		#println("NB: $nb")
		ns = getnodes(current, nb)
		#println("Nodes: $ns")
		current = ns[end]
		if length(nodes) == 0
			append!(nodes, ns)
		else
			append!(nodes, ns[2:end])
		end
		next += 1
	end

	return nodes
	#=
	p = Any[]
	println("NS: $nodes")
	for n in nodes
		t = map(x->round(Int, x), n)
		if length(p) == 0 || p[end] != t
			push!(p, t)
		end
	end
	return p
	=#
end

function getnodes(n1, n2)
	nodes = Any[]
	if n1[1] == n2[1]
		if n1[2] > n2[2]
			if n1[3] != 270
				c = n1[3]
				while c != 0
					push!(nodes, (n1[1], n1[2], c))
					c -= 90
				end
			else
				push!(nodes, n1)
			end
			push!(nodes, (n1[1], n1[2], 0))
			push!(nodes, (n2[1], n2[2], 0))
		else
			if n1[3] != 270
				c = n1[3]
				while c != 180
					push!(nodes, (n1[1], n1[2], c))
					c += 90
				end
			else
				push!(nodes, n1)

			end
			push!(nodes, (n1[1], n1[2], 180))
			push!(nodes, (n2[1], n2[2], 180))

		end
	else
		if n1[1] > n2[1]
			if n1[3] != 0
				c = n1[3]
				while c != 270
					push!(nodes, (n1[1], n1[2], c))
					c += 90
				end
			else
				push!(nodes, n1)
			end
			push!(nodes, (n1[1], n1[2], 270))
			push!(nodes, (n2[1], n2[2], 270))
		else
			if n1[3] != 0
				c = n1[3]
				while c != 90
					push!(nodes, (n1[1], n1[2], c))
					c -= 90
				end
			else
				push!(nodes, n1)

			end
			push!(nodes, (n1[1], n1[2], 90))
			push!(nodes, (n2[1], n2[2], 90))
		end
	end
	return nodes
end

function segment_path(path)
	"""
	Segment the path into forward movements and turns
	"""
	segments = Any[]

	c = 1
	curr = Any[]
	move = path[1][3] == path[2][3]

	while c < length(path)
		if path[c][3] == path[c+1][3]
			if move
				push!(curr, path[c])
				c += 1
			else
				push!(curr, path[c])
				push!(segments, curr)
				curr = Any[]
				move = true
			end
		else
			if !move
				push!(curr, path[c])
				c += 1
			else
				push!(curr, path[c])
				push!(segments, curr)
				curr = Any[]
				move = false
			end
		end
	end
	push!(curr, path[end])
	push!(segments, curr)
	return segments
end

function test()
	h,w = (6, 6)
	maze = generate_maze(h, w)
	print_maze(maze)
	path = generate_path(maze)
	
	for i=1:length(path)-1
		print(path[i])
		print(" => ")
	end
	 
        println(path[end])
	segments = segment_path(path)
	for s in segments
		println(s)
	end
end

test()
