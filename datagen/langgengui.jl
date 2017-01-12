include("maze.jl")
include("path_generator.jl")
include("lang_generator.jl")

using ThreeJS
using Compat

WallColors = Dict(1=>"orangered", 2=>"yellow", 3=>"black")
FloorColors= Dict(1=>"navy", 2=>"firebrick", 3=>"lavender",
	4=>"violet", 5=>"darkgreen", 6=>"gray", 7=>"saddlebrown", 8=>"yellowgreen")
ItemInitials = Dict(1=>"B", 2=>"C", 3=>"E", 4=>"H", 5=>"L", 6=>"S", 7=>"")

function get_node_mesh(i, j, ht, wt, item)
	ms = Any[]
	m = mesh(-270+wt*(j-1)+(wt/2), 270-ht*(i-1)-(ht/2), 0.0) << [ThreeJS.plane(wt*0.6, ht*0.6),material(Dict(:kind=>"basic", :color=>"teal"))]
	push!(ms, m)

	ratio = 0.25
	d = min(ht, wt)*0.6*ratio
	dist = 60.0/d

	if item != ""
		m = ThreeJS.text(dist*(-270+wt*(j-1)+(wt/2)+wt*0.6*0.5*ratio)/800.0,
			dist*(270-ht*(i-1)-(ht/2)+ht*0.6*0.5*ratio)/800.0, (800.0-dist), item)
		push!(ms, m)
	end
	return ms
end

#horizontal edge
function hedge_mesh(i, j, ht, wt, wall, floor)
	ms = Any[]
	m = mesh(-270+wt*(j), 270-ht*(i-1)-(ht)*0.25, 0.0) << [ThreeJS.plane(wt*0.4, ht*0.1),material(Dict(:kind=>"basic", :color=>wall))]
	push!(ms, m)
	m = mesh(-270+wt*(j), 270-ht*(i)+ht*0.25, 0.0) << [ThreeJS.plane(wt*0.4, ht*0.1),material(Dict(:kind=>"basic", :color=>wall))]
	push!(ms, m)
	m = mesh(-270+wt*(j), 270-ht*(i)+ht*0.5, 0.0) << [ThreeJS.plane(wt*0.4, ht*0.4),material(Dict(:kind=>"basic", :color=>floor))]
	push!(ms, m)
	return ms
end

#vertical edge
function vedge_mesh(i, j, ht, wt, wall, floor)
	ms = Any[]
	m = mesh(-270+wt*(j-1)+wt*0.25, 270-ht*i, 0.0) << [ThreeJS.plane(wt*0.1, ht*0.4),material(Dict(:kind=>"basic", :color=>wall))]
	push!(ms, m)
	m = mesh(-270+wt*(j)-wt*0.25, 270-ht*i, 0.0) << [ThreeJS.plane(wt*0.1, ht*0.4),material(Dict(:kind=>"basic", :color=>wall))]
	push!(ms, m)
	m = mesh(-270+wt*(j)-wt*0.5, 270-ht*i, 0.0) << [ThreeJS.plane(wt*0.4, ht*0.4),material(Dict(:kind=>"basic", :color=>floor))]
	push!(ms, m)
	return ms
end

function draw_map(map, maze, h, w)
	meshes = Any[]
	ht = 540.0 / h
	wt = 540.0 / w

	for i=1:h
		for j=1:w
			it = map.nodes[(j, i)]
			m = get_node_mesh(i, j, ht, wt, ItemInitials[it])
			append!(meshes, m)
			if maze[i, j, 2] == 1
				wall, floor = map.edges[(j, i)][(j+1, i)]
				append!(meshes, hedge_mesh(i, j, ht, wt, WallColors[wall], FloorColors[floor]))
			end

			if maze[i, j, 3] == 1
				wall, floor = map.edges[(j, i)][(j, i+1)]
				append!(meshes, vedge_mesh(i, j, ht, wt, WallColors[wall], FloorColors[floor]))
			end
		end
	end

	return meshes
end

function draw_path(path, h, w)
	ht = 540.0 / h
	wt = 540.0 / w

	ms = Any[]
	ratio = 0.2
	cs = ["crimson", "cyan", "lime"]

	for ind=1:length(path)
		i, j = path[ind]
		cname = ind == 1 ? cs[1] : ind == length(path) ? cs[3] : cs[2]
		m = mesh(-270+wt*(j-1)+(wt/2), 270-ht*(i-1)-(ht/2), 0.0) << [ThreeJS.plane(wt*0.6*ratio, ht*0.6*ratio),material(Dict(:kind=>"basic", :color=>cname))]
		push!(ms, m)
	end
	return ms
end

function draw_frame()
	meshes = Any[]
	m = mesh(0.0, 272.5, 0.0) << [ThreeJS.box(550.0, 5.0, 0.0),material(Dict(:kind=>"lambert", :color=>"black"))]
	push!(meshes, m)
	m = mesh(0.0, -272.5, 0.0) << [ThreeJS.box(550.0, 5.0, 0.0),material(Dict(:kind=>"lambert", :color=>"black"))]
	push!(meshes, m)
	m = mesh(272.5, 0.0, 0.0) << [ThreeJS.box(5.0, 550.0, 0.0),material(Dict(:kind=>"lambert", :color=>"black"))]
	push!(meshes, m)
	m = mesh(-272.5, 0.0, 0.0) << [ThreeJS.box(5.0, 550.0, 0.0),material(Dict(:kind=>"lambert", :color=>"black"))]
	push!(meshes, m)
	return meshes
end

function to_string_html(generation)
	i = 0
	hs = Any[]
	l = Any[]
	for (s, ins) in generation
		i += 1
		push!(l, plaintext(string(s)))
		push!(l, plaintext(string(ins)))

		if i > 1 && i % 5 == 0
			push!(hs, vbox(l))
			push!(hs, hskip(1em))
			push!(hs, vline())
			push!(hs, hskip(1em))
			l = Any[]
		end
	end
	if i % 5 != 0
		push!(hs, vbox(l))
		push!(hs, hskip(1em))
		push!(hs, vline())
		push!(hs, hskip(1em))
	end
	return hbox(hs)
end

function main(window)
	push!(window.assets,("ThreeJS","threejs"))
	push!(window.assets, "widgets")

	state = Dict()
	state[:maze] = nothing
	state[:navimap] = nothing
	state[:path] = nothing
	state[:nodes] = nothing
	state[:maze_meshes] = Any[]
	state[:path_meshes] = Any[]

	inp = Signal(Dict())
	inp2 = Signal(state)
	s = Escher.sampler()
	
	form = 	hbox(
		watch!(s, :h, textinput("8"; label="Height")),
		hskip(1em),
		watch!(s, :w, textinput("8"; label="Width")),
		trigger!(s, :mazebut, button("Generate maze")),
		trigger!(s, :pathbut, button("Generate path")),
		trigger!(s, :insbut, button("Generate instruction")))
		
	
	map(inp, inp2) do dict, dict2
		txt = plaintext("")
		meshes = Any[]
		
		if haskey(dict, :mazebut)
			cam = 800.0
			dict2[:maze_meshes] = Any[]
			dict2[:path_meshes] = Any[]
			dict2[:maze_meshes] = draw_frame()
			push!(dict2[:maze_meshes], ambientlight())
			push!(dict2[:maze_meshes], camera(0.0, 0.0, cam))
			h = parse(Int, dict[:h])
			w = parse(Int, dict[:w])
			dict2[:maze] = generate_maze(h, w)
			dict2[:navimap] = generate_navi_map(dict2[:maze], "langen")
			ms = draw_map(dict2[:navimap], dict2[:maze], h, w)
			append!(dict2[:maze_meshes], ms)
		end
		append!(meshes, dict2[:maze_meshes])
		
		if haskey(dict, :pathbut) && dict2[:maze] != nothing
			dict2[:path_meshes] = Any[]
			dict2[:nodes], dict2[:path] = generate_path(dict2[:maze]; distance=3)
			h,w,_ = size(dict2[:maze])
			dict2[:path_meshes] = draw_path(dict2[:path], h, w)
		end
		append!(meshes, dict2[:path_meshes])
		
		if haskey(dict, :insbut) && dict2[:nodes] != nothing
			segments = segment_path(dict2[:nodes])
			generation = generate_lang(dict2[:navimap], dict2[:maze], segments)
			txt = to_string_html(generation)
		end
		
		vbox(
		intent(s, form) >>> inp,
		ThreeJS.outerdiv() <<
		(
		ThreeJS.initscene() << meshes
		),
		hline(),
		txt
		)
	end
end
