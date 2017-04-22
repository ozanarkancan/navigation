using Logging, ArgParse

include("maze.jl")
include("path_generator.jl")
include("lang_generator.jl")

CHARS = collect("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
items = filter(x->x!="", collect(keys(Items)))
floors = collect(keys(Floors))
walls = collect(keys(Walls))

function turn_to_x(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)
        mname = join(rand(CHARS, 20))
        navimap.name = mname
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        gen = generate_lang(navimap, maze, segments; combine=0.0)

        for (s, inst) in gen
            cats = inst[2:end]
            if length(cats) == 1 && cats[1] == visual_t
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end

function move_to_x(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        mname = join(rand(CHARS, 20))
        navimap.name = mname
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        gen = generate_lang(navimap, maze, segments; combine=0.0)

        for (s, inst) in gen
            cats = inst[2:end]
            if length(cats) == 1 && cats[1] == visual_m
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end

    return ins, navimap
end

combined_12(name, id) = rand([turn_to_x, move_to_x])(name, id)

function turn_and_move_to_x(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.3)
        mname = join(rand(CHARS, 20))
        navimap.name = mname
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        gen = generate_lang(navimap, maze, segments; combine=1.0)

        for (s, inst) in gen
            cats = inst[2:end]
            if length(cats) == 1 && cats[1] == visual_tm
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end

function lang_only(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)

        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.3], iprob=0.5)
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        mname = join(rand(CHARS, 20))
        navimap.name = mname
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        gen = generate_lang(navimap, maze, segments; combine=0.5)

        for (s, inst) in gen
            cats = inst[2:end]
            langvalid = true
            for c in cats
                if !(c == langonly_t || c == langonly_m)
                    langvalid = false
                end
            end

            if langvalid
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end 
    end
    return ins, navimap
end

combined_1245(name, id) = rand([turn_to_x, move_to_x, turn_and_move_to_x, lang_only])(name, id)

function turn_to_x_and_move(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)

        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        
        mname = join(rand(CHARS, 20))
        navimap.name = mname

        gen = generate_lang(navimap, maze, segments; combine=1.0)

        for (s, inst) in gen
            cats = inst[2:end]

            if length(cats) == 2 && cats[1] == visual_t && cats[2] == langonly_m
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end 
    end
    return ins, navimap
end

function turn_move_to_x(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)

        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        
        mname = join(rand(CHARS, 20))
        navimap.name = mname

        gen = generate_lang(navimap, maze, segments; combine=1.0)

        for (s, inst) in gen
            cats = inst[2:end]

            if length(cats) == 2 && cats[1] == langonly_t && cats[2] == visual_m
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end 
    end
    return ins, navimap
end

function move_to_x_and_turn(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        mname = join(rand(CHARS, 20))
        navimap.name = mname
        gen = generate_lang(navimap, maze, segments; combine=1.0)

        for (s, inst) in gen
            cats = inst[2:end]

            if length(cats) == 2 && cats[1] == visual_m && cats[2] == langonly_t
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end

function move_turn_to_x(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        mname = join(rand(CHARS, 20))
        navimap.name = mname
        gen = generate_lang(navimap, maze, segments; combine=1.0)

        for (s, inst) in gen
            cats = inst[2:end]

            if length(cats) == 2 && cats[1] == langonly_m && cats[2] == visual_t
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end


combined_78(name, id) = rand([turn_to_x_and_move, move_to_x_and_turn])(name, id)

function turn_to_x_move_to_y(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        mname = join(rand(CHARS, 20))
        navimap.name = mname
        gen = generate_lang(navimap, maze, segments; combine=1.0)

        for (s, inst) in gen
            cats = inst[2:end]

            if length(cats) == 2 && cats[1] == visual_t && cats[2] == visual_m
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end

function move_to_x_turn_to_y(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        mname = join(rand(CHARS, 20))
        navimap.name = mname

        gen = generate_lang(navimap, maze, segments; combine=1.0)

        for (s, inst) in gen
            cats = inst[2:end]

            if length(cats) == 2 && cats[1] == visual_m && cats[2] == visual_t
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end

combined_1011(name, id) = rand([turn_to_x_move_to_y, move_to_x_turn_to_y])(name, id)
combined_781011(name, id) = rand([turn_to_x_and_move, move_to_x_and_turn, turn_to_x_move_to_y, move_to_x_turn_to_y])(name, id)
combined_45781011(name, id) = rand([turn_and_move_to_x, lang_only, turn_to_x_and_move, move_to_x_and_turn, turn_to_x_move_to_y, move_to_x_turn_to_y])(name, id)
combined_1245781011(name, id) = rand([turn_to_x_move_to_y, move_to_x_turn_to_y, turn_and_move_to_x, lang_only, turn_to_x_and_move, move_to_x_and_turn, turn_to_x_move_to_y, move_to_x_turn_to_y])(name, id)

function move_until(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        mname = join(rand(CHARS, 20))
        navimap.name = mname
        gen = generate_lang(navimap, maze, segments; combine=1.0)

        for (s, inst) in gen
            cats = inst[2:end]

            if length(cats) == 1 && cats[1] == condition_m
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end

function orient(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)
        mname = join(rand(CHARS, 20))
        navimap.name = mname
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        gen = generate_lang(navimap, maze, segments; combine=0.0)

        for (s, inst) in gen
            cats = inst[2:end]
            if length(cats) == 1 && cats[1] == orient_t
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end

function describe(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)
        mname = join(rand(CHARS, 20))
        navimap.name = mname
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        gen = generate_lang(navimap, maze, segments; combine=0.0)

        for (s, inst) in gen
            cats = inst[2:end]
            if length(cats) == 1 && cats[1] == description
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end

function turn_move_until(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        mname = join(rand(CHARS, 20))
        navimap.name = mname

        gen = generate_lang(navimap, maze, segments; combine=1.0)

        for (s, inst) in gen
            cats = inst[2:end]

            if length(cats) == 2 && cats[1] == langonly_t && cats[2] == condition_m
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end

function turn_to_x_move_until(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        mname = join(rand(CHARS, 20))
        navimap.name = mname

        gen = generate_lang(navimap, maze, segments; combine=1.0)

        for (s, inst) in gen
            cats = inst[2:end]

            if length(cats) == 2 && cats[1] == visual_t && cats[2] == condition_m
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end

function move_until_turn(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        mname = join(rand(CHARS, 20))
        navimap.name = mname

        gen = generate_lang(navimap, maze, segments; combine=1.0)

        for (s, inst) in gen
            cats = inst[2:end]

            if length(cats) == 2 && cats[1] == condition_m && cats[2] == langonly_t
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end

function move_until_turn_to_x(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.4)
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        mname = join(rand(CHARS, 20))
        navimap.name = mname

        gen = generate_lang(navimap, maze, segments; combine=1.0)

        for (s, inst) in gen
            cats = inst[2:end]

            if length(cats) == 2 && cats[1] == condition_m && cats[2] == visual_t
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end

"""
Available task functions:

t1: turn_to_x
t2: move_to_x
t3: combined_12 : generate data using turn_to_x and move_to_x
t4: turn_and_move_to_x
t5: lang_only
t6: combined_1245 : generate data using turn_to_x, move_to_x, turn_and_move_to_x, lang_only
t7: turn_to_x_and_move : the move part is lang only
t8: move_to_x_and_turn : the turn part is lang only
t9: combined_78 : generate data using turn_to_x_and_move and move_to_x_and_turn
t10: turn_to_x_move_to_y
t11: move_to_x_turn_to_y
t12: combined_1011
t13: combined_781011
t14: move_until : move until the specified condition is satisfied(condition occurs after movement start)
t15: orient : turn so that ...
t16: describe : describes the final position
t17: turn_move_to_x
t18: move_turn_to_x

"""

function generatedata(taskf; numins=100)
    data = Any[]
    mps = Dict()
    
    inscount = 0
    while inscount < numins
        inscount += 1
        name = string("artificial_", inscount)

        ins, mp = taskf(name, inscount)
        push!(data, ins)
        mps[mp.name] = mp
    end

    return data, mps
end

function parse_commandline(argv)
    isa(argv, AbstractString) && (argv=split(argv))
    s = ArgParseSettings()

    @add_arg_table s begin
	("--log"; help = "name of the log file"; default = "test.log")
    end
    return parse_args(argv,s)
end

function testgeneratedata(argv=ARGS)
    args = parse_commandline(argv)
    Logging.configure(filename=args["log"])
    Logging.configure(level=INFO)
    
    for i=1:100
        info("i: $i")
        dat, maps = generatedata(turnToX)
        for ins in dat
            m = maps[ins.map]
            fs = filter(t->m.nodes[t] != 7, collect(keys(m.nodes)))
            @assert length(fs) != 0
            its = map(t->m.nodes[t], fs)

            @assert length(its) == length(Set(its))
        end
    end

    info("PASSED")
end

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

#testgeneratedata()
