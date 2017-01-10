include("maze.jl")

using ThreeJS
using Compat

WallColors = Dict(1=>"orangered", 2=>"yellow", 3=>"black")
FloorColors= Dict(1=>"navy", 2=>"firebrick", 3=>"lavender",
	4=>"violet", 5=>"darkgreen", 6=>"gray", 7=>"saddlebrown", 8=>"yellowgreen")
ItemInitials = Dict(1=>"B", 2=>"C", 3=>"E", 4=>"H", 5=>"L", 6=>"S", 7=>"")

function get_node_mesh(i, j, ht, wt, item)
	println("i,j,item: $((i,j,item))")
	ms = Any[]
	m = mesh(-270+wt*(j-1)+(wt/2), 270-ht*(i-1)-(ht/2), 0.0) << [ThreeJS.plane(wt*0.6, ht*0.6),material(Dict(:kind=>"basic", :color=>"teal"))]
	push!(ms, m)

	d = min(ht, wt)*0.6*0.25
	dist = 60.0/d

	println("Dist: $dist")

	if item != ""
		m = ThreeJS.text(dist*(-270+wt*(j-1)+(wt/2))/800.0, dist*(270-ht*(i-1)-(ht/2))/800.0, (800.0-dist), item)
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

function main(window)
	push!(window.assets,("ThreeJS","threejs"))
	push!(window.assets, "widgets")

	inp = Signal(Dict())
	s = Escher.sampler()
	
	form = 	hbox(
		watch!(s, :h, textinput("3"; label="Height")),
		hskip(1em),
		watch!(s, :w, textinput("3"; label="Width")),
		trigger!(s, :submit, button("Generate maze")))
		
	
	map(inp) do dict
		txt = "No text."
		cam = 800.0
		meshes = draw_frame()
		push!(meshes, ambientlight())
		push!(meshes, camera(0.0, 0.0, cam))
		#push!(meshes, mesh(0.0, 0.0, 0.0) << [ThreeJS.box(60.0, 60.0, 0.0),material(Dict(:kind=>"lambert", :color=>"teal"))])
		#push!(meshes, ThreeJS.text(270/800.0, 0.0, 799.0, "E"))
		if haskey(dict, :h)
			txt = string(dict[:h], " x ", dict[:w])
			h = parse(Int, dict[:h])
			w = parse(Int, dict[:w])
			maze = generate_maze(h, w)
			navimap = generate_navi_map(maze, "langen")
			print_maze(maze)
			ms = draw_map(navimap, maze, h, w)
			append!(meshes, ms)
		end

		vbox(
		intent(s, form) >>> inp,
		hbox(
		ThreeJS.outerdiv() <<
		(
		ThreeJS.initscene() << meshes
		),
		plaintext(txt))
		)
	end
end
