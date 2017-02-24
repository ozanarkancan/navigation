using ArgParse

include("util.jl")
include("io.jl")

function parse_commandline(argv)
    isa(argv, AbstractString) && (argv=split(argv))
    s = ArgParseSettings()

    @add_arg_table s begin
	"--hidden"
	help = "hidden size"
	default = 100
	arg_type = Int
	"--embed"
	help = "embedding size"
	default = 100
	arg_type = Int
	"--limactions"
	arg_type = Int
	default = 35
	"--trainfiles"
	help = "built training jld file"
	default = ["grid_jelly.jld", "grid_l.jld", "l_jelly.jld"]
	nargs = '+'
	"--testfiles"
	help = "test file as regular instruction file(json)"
	default = ["l",
		   "jelly",
		   "grid"]
	nargs = '+'
	"--window"
	help = "size of the filter"
	default = [29, 7, 5]
	arg_type = Int
	nargs = '+'
	"--filters"
	help = "number of filters"
	default = [300, 150, 50]
	arg_type = Int
	nargs = '+'
	"--model"
	help = "model file"
	default = "baseline_cnn_wvecs.jl"
	"--pdrops"
	help = "dropout rates"
	nargs = '+'
	default = [0.2, 0.5, 0.5]
	arg_type = Float64
	"--pdrops_dec"
	help = "dropout rates"
	nargs = '+'
	default = [0.2, 0.5, 0.5]
	arg_type = Float64
	"--bs"
	help = "batch size"
	default = 1
	arg_type = Int
	"--log"
	help = "name of the log file"
	default = "test.log"
	"--test"
	help = "1,2 or 3 (l, jelly, grid)"
	arg_type = Int
	"--load"
	help = "model path"
	default = ["mbank/cnn_wvecs_3_1_l_jelly.jld"]
	nargs = '+'
	"--charenc"
	help = "charecter embedding"
	action = :store_true
	"--encoding"
	help = "grid or multihot"
	default = "grid"
	"--greedy"
	help = "deterministic or stochastic policy"
	action = :store_false
	"--embedding"
	help = "embedding"
	action = :store_false
	"--seed"
	help = "seed"
	default = 0
	arg_type = Int
        "--mapname"
        help = "name of the map"
        default = "Grid"
    end
    return parse_args(argv,s)
end		

function get_maps()
    fname = "data/maps/map-grid.json"
    grid = getmap(fname)

    fname = "data/maps/map-jelly.json"
    jelly = getmap(fname)

    fname = "data/maps/map-l.json"
    l = getmap(fname)

    maps = Dict("Grid" => grid, "Jelly" => jelly, "L" => l)
    return maps
end

function main(argv=ARGS)
    global grid,jelly,l         # Vector{Instruction}
    global maps                 # Dict{Grid|L|Jelly,Map}
    global vocab                # Dict{WordStr,Int}
    global emb                  # Dict{WordStr,Vector{Float32}(300)}
    global models               # Vector{Model}(1)
    args = parse_commandline(argv)
    include(args["model"])
    if args["seed"]>0; srand(args["seed"]); end

    Logging.configure(output=STDOUT)
    Logging.configure(level=INFO)
    info("*** Parameters ***")
    for k in keys(args); info("$k -> $(args[k])"); end

    info("Loading...")
    grid, jelly, l = getallinstructions()

    maps = get_maps()

    vocab = !args["charenc"] ? build_dict(vcat(grid, jelly, l)) : build_char_dict(voc_ins)
    info("\nVocab: $(length(vocab))")
    emb = load("data/embeddings.jld", "vectors")

    models = Any[]
    for mfile in args["load"]; push!(models, loadmodel(mfile)); end

    info("Working on $(args["mapname"]) map...")
    map1 = maps[args["mapname"]]
    x,y = collect(keys(map1.nodes))[rand(1:length(map1.nodes))]
    o = rand(0:90:270)

    while true
        try
            # print("Coordinates and the orientation(x,y,o): ")
            # str = readline(STDIN)
            # x,y,o = map(x->parse(Int, x), split(strip(str), ","))

            println(mapstring(map1, (x,y,o)))
            print("Enter an instruction: ")
            str = strip(readline(STDIN))
            if str=="" break; end
            text = split(str)

            ins = Instruction("demo", text, Any[(x,y,o)], args["mapname"], 0)
            dat = args["embedding"] ? [(ins, ins_arr_embed(emb, vocab, ins.text))] : [(ins, ins_arr(vocab, ins.text))]

            if text[1] == "reset"
                (x,y,o) = eval(parse(text[2]))
            elseif text[1] == "quit"
                break
            else
                (x,y,o) = demotest(models, dat, maps; args=args)
            end

            # print("Continue y/n: ")
            # c = readline(STDIN)
            # c = strip(c)
            # if c=="N" || c=="n" || c=="no" || c=="No"
            #     break
            # end
        catch e
            info(e)
            info("Bad things happened...\n")
        end
    end
end

function mapstring(m::Map, coor=nothing)
    xmin,xmax = extrema(map(c->c[1],keys(m.nodes)))
    ymin,ymax = extrema(map(c->c[2],keys(m.nodes)))
    const mapchars = collect("BCEHLS.123brcfgvwy")
    const mapwords = split("barstool chair easel hatrack lamp sofa X butterfly fish tower blue brick concrete flower grass gravel wood yellow")
    maplegend = "B:barstool C:chair E:easel H:hatrack L:lamp S:sofa .:X 1:butterfly 2:fish\n3:tower b:blue r:brick c:concrete f:flower g:grass v:gravel w:wood y:yellow\nxmap:$((xmin,xmax)) ymap:$((ymin,ymax))\n"
    itemchar(i)=mapchars[i]
    wallchar(i)=mapchars[i+7]
    floorchar(i)=mapchars[i+10]
    agentchar(o) = (o==0 ? '^' : o==90 ? '>' : o==180 ? 'V' : o==270 ? '<' : '?')
    ncols = 3+5*(xmax-xmin)
    itemcoor(x,y) = (3*(y-ymin)*ncols + 5*(x-xmin) + 1)
    wallcoor(x1,y1,x2,y2) = (x1==x2 ? itemcoor(x1,min(y1,y2))+2*ncols : itemcoor(min(x1,x2),y1)+3)
    floorcoor(x1,y1,x2,y2) = (x1==x2 ? itemcoor(x1,min(y1,y2))+ncols : itemcoor(min(x1,x2),y1)+2)
    chars = fill!(Array(Char,itemcoor(xmax,ymax)+1),' ')
    chars[ncols:ncols:end] = '\n'
    for (xy,item) in m.nodes
        chars[itemcoor(xy...)] = itemchar(item)
    end
    for (xy1,neighbors) in m.edges
        for (xy2,wallfloor) in neighbors
            w,f = wallfloor
            chars[wallcoor(xy1...,xy2...)] = wallchar(w)
            chars[floorcoor(xy1...,xy2...)] = floorchar(f)
        end
    end
    if coor != nothing
        x,y,o = coor
        chars[itemcoor(x,y)] = agentchar(o)
        maplegend *= "Agent($(agentchar(o))) at $coor\n"
    end
    return maplegend*"\n"*String(chars)
end

function demotest(models, data, maps; args=nothing)
    current = actions = nothing
    scss = 0.0
    mask = convert(KnetArray, ones(Float32, 1,1))

    for (instruction, words) in data
	words = map(v->convert(KnetArray{Float32},v), words)
	states = map(weights->initstate(KnetArray{Float32}, convert(Int, size(weights["enc_b1_f"],2)/4), 1, length(words)), models)

	for ind=1:length(models)
	    weights = models[ind]
	    state = states[ind]
	    encode(weights["enc_w1_f"], weights["enc_b1_f"], weights["enc_w1_b"], weights["enc_b1_b"], weights["emb_word"], state, words)
	    
	    state[5] = hcat(state[1][end], state[3][end])
	    state[6] = hcat(state[2][end], state[4][end])
	end
	
	current = instruction.path[1]
	nactions = 0
	stop = false
	
	actions = Any[]

	while !stop
	    view = state_agent_centric(maps[instruction.map], current)
	    view = convert(KnetArray{Float32}, view)
	    cum_ps = zeros(Float32, 1, 4)
	    for ind=1:length(models)
		weights = models[ind]
		state = states[ind]
		x = spatial(weights["filters_w1"], weights["filters_b1"], weights["filters_w2"], weights["filters_b2"], 
			    weights["filters_w3"], weights["filters_b3"], weights["emb_world"], view)

		ypred = decode(weights["dec_w1"], weights["dec_b1"], weights["soft_w1"], weights["soft_w2"], 
			       weights["soft_w3"], weights["soft_b"], state, x, mask)
		cum_ps += probs(Array(ypred))
	    end

	    cum_ps = cum_ps ./ length(models)
	    # info("Probs: $(cum_ps)")
	    action = 0
	    if args["greedy"]
		action = indmax(cum_ps)
	    else
		action = sample(cum_ps)
	    end
	    
	    push!(actions, action)
	    prev = current
	    current = getlocation(maps[instruction.map], current, action)
	    nactions += 1

	    nowall = false
	    if action == 1
		nowall = !haskey(maps[instruction.map].edges[(prev[1], prev[2])], (current[1], current[2]))
	    end

	    stop = nactions > args["limactions"] || action == 4 || nowall
	    
	end
	
	# info("$(instruction.text)")
	# info("Path: $(instruction.path)")
	# info("Filename: $(instruction.fname)")

	# info("Actions: $(reshape(collect(actions), 1, length(actions)))")
	# info("Current: $(current)")

	if current == instruction.path[end]
	    scss += 1
	    # info("SUCCESS\n")
	else
	    # info("FAILURE\n")
	end
    end

    # return scss / length(data)
    const actwords = split("move right left stop")
    actstr = join(actwords[actions], " ")
    info("Actions: $actstr")
    return current
end

# main()
