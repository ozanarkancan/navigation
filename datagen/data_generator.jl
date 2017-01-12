include("maze.jl")
include("path_generator.jl")
include("lang_generator.jl")

function test()
	h,w = (6, 6)
	maze = generate_maze(h, w)
	print_maze(maze)
	nodes, path = generate_path(maze; distance=3)
	
	for i=1:length(nodes)-1
		print(nodes[i])
		print(" => ")
	end
        println(nodes[end])

	segments = segment_path(nodes)
	println("Segments:")
	for s in segments
		println(s)
	end

	navimap = generate_navi_map(maze, "123")

	generation = generate_lang(navimap, maze, segments)

	for (s,ins) in generation
		println((s, ins))
	end

	#=
	map = generate_navi_map(maze, "123")

	
	for k in keys(map.nodes)
		println("Node: $k , item: $(map.nodes[k])")
	end

	for n1 in keys(map.edges)
		for n2 in keys(map.edges[n1])
			println("$n1 <-> $n2 : $(map.edges[n1][n2])")
		end
	end
	=#
end

test()
