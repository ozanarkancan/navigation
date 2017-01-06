using Base.Collections

#recursive backtracking algorithm
function generate_maze(h = 4, w = 4)
	maze = zeros(h, w, 4)
	unvisited = ones(h, w)

	function neighbours(r,c)
		ns = Array{Tuple{Int, Int, Int}, 1}()
		for i=1:4
			if i == 1 && (r - 1) >= 1 && unvisited[r-1, c] == 1
				push!(ns, (r-1, c, 1))
			elseif i == 2 && (c + 1) <= w && unvisited[r, c+1] == 1
				push!(ns, (r, c+1, 2))
			elseif i == 3 && (r+1) <= h && unvisited[r+1, c] == 1
				push!(ns, (r+1, c, 3))
			elseif i == 4 && (c-1) >= 1 && unvisited[r, c-1] == 1
				push!(ns, (r, c-1, 4))
			end
		end
		return shuffle(ns)
	end

	start = rand(1:h*w)
	r = div(start, h) + 1
	r = r > h ? h : r
	c = start % w
	c = c == 0 ? w : c

	stack = Array{Tuple{Int, Int}, 1}()
	curr = (r, c)
	unvisited[r,c] = 0
	while countnz(unvisited) != 0
		ns = neighbours(curr[1], curr[2])
		if length(ns) > 0
			r,c = curr
			rn, cn, d = ns[1]
			push!(stack, (r,c))
			maze[r, c, d] = 1
			dn = d - 2 <= 0 ? d + 2 : d - 2
			maze[rn, cn, dn] = 1
			curr = (rn, cn)
			unvisited[rn, cn] = 0
		elseif length(stack) != 0
			curr = pop!(stack)
		end

	end
	return maze
end

function print_maze(maze)
	h,w,_ = size(maze)
	rows = 2*h + 1
	cols = 2*w + 1

	for i=1:rows
		println("")
		for j=1:cols
			if i == 1 || i == rows || j == 1 || j == cols
				print("#")
			elseif i % 2 == 1 && j % 2 == 1
				print("#")
			elseif i % 2 == 1 && j % 2 == 0
				r = div(i - 1, 2)
				c = div(j, 2)
				if maze[r, c, 3] == 1
					print(" ")
				else
					print("#")
				end
			elseif i % 2 == 0 && j % 2 == 1
				r = div(i, 2)
				c = div(j - 1, 2)
				if maze[r,c,2] == 1
					print(" ")
				else
					print("#")
				end
			else
				print(" ")
			end

		end
	end
	print("\n")
end

#start & goal must be an array with 2 elements
function astar_solver(maze, start, goal)
	function neighbours(r,c)
		ns = Any[]
		for i=1:4
			if i == 1 && maze[r, c, 1] == 1
				push!(ns, Float64[r-1, c])
			elseif i == 2 && maze[r, c, 2] == 1
				push!(ns, Float64[r, c+1])
			elseif i == 3 && maze[r, c, 3] == 1
				push!(ns, Float64[r+1, c])
			elseif i == 4 && maze[r, c, 4] == 1
				push!(ns, Float64[r, c-1])
			end
		end
		return ns
	end

	closed = Set()
	open = PriorityQueue{Array{Float64, 1}, Float64, Base.Order.ForwardOrdering}()

	parent = Dict()
	path_cost = Dict()
	heuristic = Dict()

	path_cost[start] = 0.0
	heuristic[start] = norm(start - goal)
	parent[start] = [0.0, 0.0]

	Collections.enqueue!(open, start, path_cost[start] + heuristic[start])

	current = nothing

	while length(open) != 0
		current = Collections.dequeue!(open)

		if current == goal; break; end

		push!(closed, current)

		ns = neighbours(convert(Int, current[1]), convert(Int, current[2]))
		for n in ns
			if n in closed; continue; end
			if n in keys(open)
				if path_cost[n] > path_cost[current] + 1
					path_cost[n] = path_cost[current] + 1
					heuristic[n] = norm(n - goal)
					open[n] = path_cost[n] + heuristic[n]
					parent[n] = current
				end
			else
				path_cost[n] = path_cost[current] + 1
				heuristic[n] = norm(n- goal)
				Collections.enqueue!(open, n, path_cost[n] + heuristic[n])
				parent[n] = current
			end
		end
	end

	path = Any[]

	while current != [0.0, 0.0]
		push!(path, current)
		current = parent[current]
	end

	return reverse(path)
end

function testmazepath()
	h,w=(10, 10)
	maze = generate_maze(h, w)
	print_maze(maze)

	start = [1.0, 1.0]
	goal = [4.0, 6.0]

	path = astar_solver(maze, start, goal)

	for i=1:length(path)-1
		print(path[i])
		print(" => ")
	end

	println(path[end])
end

#testmazepath()
