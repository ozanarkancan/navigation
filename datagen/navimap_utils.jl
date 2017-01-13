facing_wall(maze, p) = maze[p[1], p[2], p[3]] == 0
is_intersection(maze, p) = sum(maze[p[1], p[2], :]) >= 3
is_deadend(maze, p) = sum(maze[p[1], p[2], :]) == 1

function is_corner(maze, p)
	if sum(maze[p[1], p[2], :]) == 2
		conds = false
		for i=1:4
			if i == 4
				conds = conds || (maze[p[1], p[2], 4] == 1 && maze[p[1], p[2], 1] == 1)
			else
				conds = conds || (maze[p[1], p[2], i] == 1 && maze[p[1], p[2], i] == 1)
			end
		end
		return conds
	else
		return false
	end
end

function around_different_walls_floor(navimap, node)
	ws = Set()
	fs = Set()
	for n2 in keys(navimap.edges[node])
		w, f = navimap.edges[node][n2]
		push!(ws, w)
		push!(fs, f)
	end
	return (length(ws) == length(keys(navimap.edges[node])), length(fs) == length(keys(navimap.edges[node])))
end
