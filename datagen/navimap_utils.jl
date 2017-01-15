#=
TODO

check floor pattern with the turn direction
floor pattern intersection
=#

facing_wall(maze, p) = maze[p[1], p[2], p[3]] == 0
is_intersection(maze, p) = sum(maze[p[1], p[2], :]) >= 3
is_deadend(maze, p) = sum(maze[p[1], p[2], :]) == 1
rightof(d) = ((d+1) % 4) == 0 ? 4 : ((d+1) % 4)
leftof(d) = (((d-1)+4) % 4) == 0 ? 4 : (((d-1)+4) % 4)
backof(d) = ((d+2) % 4) == 0 ? 4 : ((d+2) % 4)

function is_corner(maze, p)
	if sum(maze[p[1], p[2], :]) == 2
		conds = false
		for i=1:4
			if i == 4
				conds = conds || (maze[p[1], p[2], 4] == 1 && maze[p[1], p[2], 1] == 1)
			else
				conds = conds || (maze[p[1], p[2], i] == 1 && maze[p[1], p[2], i+1] == 1)
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

function count_alleys(maze, segment)
	count = 0
	for i=2:length(segment)
		p = (segment[i][2], segment[i][1], -1)
		count += is_intersection(maze, p) ? 1 : 0
	end

	return count
end

function item_single_on_this_segment(navimap, segment)
	cnt = 0
	item = navimap.nodes[(segment[end][1], segment[end][2])]
	for s in segment
		it = navimap.nodes[(s[1], s[2])]
		cnt += it == item ? 1 : 0
	end

	return cnt == 1
end
