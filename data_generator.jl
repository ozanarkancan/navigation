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
