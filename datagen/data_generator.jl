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

        r = rand(1:3)#1:item, #2:floor, #3:wall
        if r == 1
            navimap = generate_navi_map(maze, ""; iprob=-1.0)
        else
            navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.3)
        end

        mname = join(rand(CHARS, 20))
        navimap.name = mname

        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        gen = generate_lang(navimap, maze, segments; combine=0.0)

        for (s, inst) in gen
            println(typeof(inst))
            println(inst)
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

        r = rand(1:4)#1:item or end/wall/segment/intersection
        if r == 1
            navimap = generate_navi_map(maze, ""; iprob=-1.0)
        else
            navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.3)
        end

        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)

        mname = join(rand(CHARS, 20))
        navimap.name = mname

        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        gen = generate_lang(navimap, maze, segments; combine=0.0)

        for (s, inst) in gen
            println(typeof(inst))
            println(inst)
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

        r = rand(1:3)#1:item, #2:floor, #3:wall
        if r == 1
            navimap = generate_navi_map(maze, ""; iprob=-1.0)
        else
            navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.3)
        end

        mname = join(rand(CHARS, 20))
        navimap.name = mname

        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        gen = generate_lang(navimap, maze, segments; combine=1.0)

        for (s, inst) in gen
            println(typeof(inst))
            println(inst)
            cats = inst[2:end]
            if length(cats) == 1 && cats[1] == visual_tm
                ins = Instruction(name, split(inst[1]), s, mname, id)
                break
            end
        end
    end
    return ins, navimap
end

function turncands(p)
    steps = length(p)-1
    a = action(p[1], p[2])
    cands = Any[]
    d = a == 2 ? "right" : "left"
    if steps == 1
        push!(cands, string("turn ", d))
        push!(cands, string("turn ", d, " one time"))
        push!(cands, string("turn ", d, " once"))
    else
        push!(cands, string("turn ", d, " two times"))
        push!(cands, string("turn ", d, " twice"))
    end
    return cands
end

function movecands(p)
    steps = length(p)-1
    cands = Any[]
    for prfx in ["move ", "go ", "walk "]
        for d in ["","forward "]
            for t in numbers[steps]
                for b in (steps == 1 ? [" step", " segment", " block"] : [" steps", " segments", " blocks"])
                    push!(cands, string(prfx, d, t, b))
                end
            end
        end
    end
    return cands
end

function lang_only(name, id)
    h,w = (8,8)
    ins = nothing
    navimap = nothing

    while ins == nothing
        maze, available = generate_maze(h, w; numdel=1)

        r = rand(1:4)#1:turn, move, turn and move, move and turn
        navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.3], iprob=0.4)

        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        
        mname = join(rand(CHARS, 20))
        navimap.name = mname

        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        gen = generate_lang(navimap, maze, segments; combine=0.4)

        for (s, inst) in gen
            println(typeof(inst))
            println(inst)
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

        r = rand(1:3)#1:item, #2:floor, #3:wall
        if r == 1
            navimap = generate_navi_map(maze, ""; iprob=-1.0)
        else
            navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.3)
        end

        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        prefx = "turn to the "
        
        for i=1:(length(segments)-1)
            tp, path = segments[i]
            if tp == "move"
                continue
            end

            mname = join(rand(CHARS, 20))
            navimap.name = mname
            nbs = nodes_visible(navimap, path[1][1:2])#Array of arrays
            tid = 0
            tl = getlocation(navimap, path[end], 1)#target location
            for j=1:length(nbs)
                if nbs[j][1] == tl[1:2]
                    tid = j
                end
            end
            mcands = movecands(segments[i+1][2])
            move_s = rand(mcands)
            if r == 1
                sits = shuffle(items)
                n1 = path[1]

                navimap.nodes[n1[1:2]] = Items[rand(sits[2:end])]#put an item on the current node
                
                for j=1:length(nbs)
                    curr = path[1][1:2]
                    trg = 0
                    if j == tid
                        trg = rand(1:length(nbs[j]))
                        navimap.nodes[nbs[j][trg]] = Items[sits[1]]
                    end
                    for nb in nbs[j]
                        if j == tid && nb == nbs[j][trg]
                            continue
                        end
                        navimap.nodes[nb] = Items[rand(sits[2:end])]
                    end
                end
                append!(path, segments[i+1][2][2:end])
                ins = Instruction(name, split(string(prefx, sits[1], " and ", move_s)), path, mname, id)
            else 
                if r == 2#floor
                    sflrs = shuffle(floors)
                    for j=1:length(nbs)
                        curr = path[1][1:2]
                        for nb in nbs[j]
                            w,f = navimap.edges[curr][nb]
                            navimap.edges[curr][nb] = (w, Floors[sflrs[j]])
                            navimap.edges[nb][curr] = (w, Floors[sflrs[j]])
                            curr = nb
                        end
                    end
                    append!(path, segments[i+1][2][2:end])
                    ins = Instruction(name, split(string(prefx, sflrs[tid], " and ", move_s)), path, mname, id)
                elseif r == 3#wall
                    swlls = shuffle(walls)
                    for j=1:length(nbs)
                        curr = path[1][1:2]
                        for nb in nbs[j]
                            w,f = navimap.edges[curr][nb]
                            navimap.edges[curr][nb] = j == tid ? (Walls[swlls[1]], f) : (Walls[swlls[rand(2:3)]], f)
                            navimap.edges[nb][curr] = j == tid ? (Walls[swlls[1]], f) : (Walls[swlls[rand(2:3)]], f)
                            curr = nb
                        end
                    end
                    append!(path, segments[i+1][2][2:end])
                    ins = Instruction(name, split(string(prefx, swlls[1], " and ", move_s)), path, mname, id)
                end
            end
            break
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

        r = rand(1:4)#1:item or end/wall/segment/intersection
        if r == 1
            navimap = generate_navi_map(maze, ""; iprob=-1.0)
        else
            navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.3)
        end

        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        prefx = "move to the "
        
        for i=1:(length(segments)-1)
            tp, path = segments[i]
            if tp == "turn"
                continue
            end

            mname = join(rand(CHARS, 20))
            navimap.name = mname
            nbs = nodes_visible(navimap, path[1][1:2])#Array of arrays
            tid = 0
            tl = path[end]#target location
            for j=1:length(nbs)
                for nb in nbs[j]
                    if nb == tl[1:2]
                        tid = j
                        break
                    end
                end
            end
            
            endpoint = map(x->round(Int, x), path[end])
            d = round(Int, endpoint[3] / 90 + 1)
            steps = length(path)-1
            
            if (r > 1 && r < 4 && !facing_wall(maze, (endpoint[2], endpoint[1], d)) || (r == 4 && steps > 4))
                continue
            end

            tcands = turncands(segments[i+1][2])
            turn_s = rand(tcands)
            if r == 1
                sits = shuffle(items)
                n1 = path[1]

                navimap.nodes[n1[1:2]] = Items[rand(sits[2:end])]#put an item on the current node
                
                for j=1:length(nbs)
                    curr = path[1][1:2]
                    if j == tid
                        navimap.nodes[tl[1:2]] = Items[sits[1]]
                    end
                    for nb in nbs[j]
                        if j == tid && nb == tl[1:2]
                            continue
                        end
                        navimap.nodes[nb] = Items[rand(sits[2:end])]
                    end
                end
                
                append!(path, segments[i+1][2][2:end])
                ins = Instruction(name, split(string(prefx, sits[1], " and ", turn_s)), path, mname, id)
            elseif r < 4
                suffx = r == 2 ? "end" : "wall"
                append!(path, segments[i+1][2][2:end])
                ins = Instruction(name, split(string(prefx, suffx, " and ", turn_s)), path, mname, id)
            else
                ord = steps == 1 ? rand(["next", "first"]) : ordinals[steps]
                suffx = " segment"
                
                append!(path, segments[i+1][2][2:end])
                ins = Instruction(name, split(string(prefx, ord, suffx, " and ", turn_s)), path, mname, id)
            end
            break
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

        r1 = rand(1:3)#1:item, #2:floor, #3:wall
        r2 = rand(1:4)#1:item or end/wall/segment/intersection
        if r1 == 1 || r2 == 1
            navimap = generate_navi_map(maze, ""; iprob=-1.0)
        else
            navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.3)
        end

        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        prefx = "turn to the "
        
        for i=1:(length(segments)-1)
            tp, path = segments[i]
            if tp == "move"
                continue
            end

            mname = join(rand(CHARS, 20))
            navimap.name = mname
            nbs = nodes_visible(navimap, path[1][1:2])#Array of arrays
            nbs2 = nodes_visible(navimap, segments[i+1][2][1][1:2])#Array of arrays
            sits = shuffle(items)
            tid = 0
            turntarg = nothing
            tl = getlocation(navimap, path[end], 1)#target location
            for j=1:length(nbs)
                if nbs[j][1] == tl[1:2]
                    tid = j
                end
            end

            tlf = segments[i+1][2][end]
            tidf = 0
            for j=1:length(nbs2)
                for nb in nbs2[j]
                    if nb == tlf[1:2]
                        tidf = j
                        break
                    end
                end
            end

            endpoint = map(x->round(Int, x), segments[i+1][2][end])
            d = round(Int, endpoint[3] / 90 + 1)
            steps = length(segments[i+1][2])-1

            if (r2 > 1 && r2 < 4 && !facing_wall(maze, (endpoint[2], endpoint[1], d)) || (r2 == 4 && steps > 4))
                continue
            end

            if r1 == 1
                n1 = path[1]

                navimap.nodes[n1[1:2]] = Items[rand(sits[3:end])]#put an item on the current node
                
                for j=1:length(nbs)
                    curr = path[1][1:2]
                    trg = 0
                    if j == tid
                        trg = rand(1:length(nbs[j]))
                        navimap.nodes[nbs[j][trg]] = Items[sits[1]]
                        tl = nbs[j][trg]
                    end
                    for nb in nbs[j]
                        if j == tid && nb == nbs[j][trg]
                            continue
                        end
                        navimap.nodes[nb] = Items[rand(sits[3:end])]
                    end
                end
                prefx = string(prefx, sits[1])
            else 
                if r1 == 2#floor
                    sflrs = shuffle(floors)
                    for j=1:length(nbs)
                        curr = path[1][1:2]
                        for nb in nbs[j]
                            w,f = navimap.edges[curr][nb]
                            navimap.edges[curr][nb] = (w, Floors[sflrs[j]])
                            navimap.edges[nb][curr] = (w, Floors[sflrs[j]])
                            curr = nb
                        end
                    end
                    prefx = string(prefx, sflrs[tid])
                elseif r1 == 3#wall
                    swlls = shuffle(walls)
                    for j=1:length(nbs)
                        curr = path[1][1:2]
                        for nb in nbs[j]
                            w,f = navimap.edges[curr][nb]
                            navimap.edges[curr][nb] = j == tid ? (Walls[swlls[1]], f) : (Walls[swlls[rand(2:3)]], f)
                            navimap.edges[nb][curr] = j == tid ? (Walls[swlls[1]], f) : (Walls[swlls[rand(2:3)]], f)
                            curr = nb
                        end
                    end
                    prefx = string(prefx, swlls[1])
                end
            end

            append!(path, segments[i+1][2][2:end])
            prefx = string(prefx, " and move to the ")
            if r2 == 1
                n1 = segments[i+1][2][1]

                for j=1:length(nbs2)
                    curr = segments[i+1][2][1][1:2]
                    if j == tidf
                        if tl[1:2] == tlf[1:2]
                            navimap.nodes[tlf[1:2]] = Items[sits[1]]
                        else
                            navimap.nodes[tlf[1:2]] = Items[sits[2]]
                        end
                    end
                    for nb in nbs2[j]
                        if (j == tidf && nb == tlf[1:2]) || nb == tl[1:2]
                            continue
                        end
                        navimap.nodes[nb] = Items[rand(sits[3:end])]
                    end
                end

                prefx = tl[1:2] == tlf[1:2] ? string(prefx, sits[1]) : string(prefx, sits[2])
                ins = Instruction(name, split(prefx), path, mname, id)
            elseif r2 < 4
                suffx = r2 == 2 ? "end" : "wall"
                ins = Instruction(name, split(string(prefx, suffx)), path, mname, id)
            else
                ord = steps == 1 ? rand(["next", "first"]) : ordinals[steps]
                suffx = " segment"
                ins = Instruction(name, split(string(prefx, ord, suffx)), path, mname, id)
            end
            break
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

        r1 = rand(1:3)#1:item, #2:floor, #3:wall
        r2 = rand(1:4)#1:item or end/wall/segment/intersection
        
        if r1 == 1 || r2 == 1
            navimap = generate_navi_map(maze, ""; iprob=-1.0)
        else
            navimap = generate_navi_map(maze, ""; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.3)
        end
        
        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        prefx = "move to the "
        
        for i=1:(length(segments)-1)
            tp, path = segments[i]
            if tp == "turn"
                continue
            end

            mname = join(rand(CHARS, 20))
            navimap.name = mname
            nbs = nodes_visible(navimap, path[1][1:2])#Array of arrays
            tid = 0
            tlm = path[end]#target location
            for j=1:length(nbs)
                for nb in nbs[j]
                    if nb == tlm[1:2]
                        tid = j
                        break
                    end
                end
            end
            
            endpoint = map(x->round(Int, x), path[end])
            d = round(Int, endpoint[3] / 90 + 1)
            steps = length(path)-1
            sits = shuffle(items)
            
            if (r2 > 1 && r2 < 4 && !facing_wall(maze, (endpoint[2], endpoint[1], d)) || (r2 == 4 && steps > 4))
                continue
            end

            if r2 == 1
                n1 = path[1]

                navimap.nodes[n1[1:2]] = Items[rand(sits[3:end])]#put an item on the current node
                
                for j=1:length(nbs)
                    curr = path[1][1:2]
                    if j == tid
                        navimap.nodes[tlm[1:2]] = Items[sits[1]]
                    end
                    for nb in nbs[j]
                        if j == tid && nb == tlm[1:2]
                            continue
                        end
                        navimap.nodes[nb] = Items[rand(sits[3:end])]
                    end
                end
                prefx = string(prefx, sits[1])
            elseif r2 < 4
                suffx = r2 == 2 ? "end" : "wall"
                prefx = string(prefx, suffx)
            else
                ord = steps == 1 ? rand(["next", "first"]) : ordinals[steps]
                suffx = " segment"
                prefx = string(prefx, ord, suffx)
            end
            
            append!(path, segments[i+1][2][2:end])
            prefx = string(prefx, " and turn to the ")
            
            nbs = nodes_visible(navimap, segments[i+1][2][1][1:2])#Array of arrays
            tid = 0
            tl = getlocation(navimap, segments[i+1][2][end], 1)#target location
            for j=1:length(nbs)
                if nbs[j][1] == tl[1:2]
                    tid = j
                end
            end

            if r1 == 1
                for j=1:length(nbs)
                    curr = segments[i+1][2][1][1:2]
                    trg = 0
                    if j == tid
                        trg = rand(1:length(nbs[j]))
                        navimap.nodes[nbs[j][trg]] = Items[sits[2]]
                        tl = nbs[j][trg]
                    end
                    for nb in nbs[j]
                        if (j == tid && nb == nbs[j][trg]) || nb[1:2] == tlm[1:2]
                            continue
                        end
                        navimap.nodes[nb] = Items[rand(sits[3:end])]
                    end
                end
                prefx = string(prefx, sits[2])
            else 
                if r1 == 2#floor
                    sflrs = shuffle(floors)
                    for j=1:length(nbs)
                        curr = segments[i+1][2][1][1:2]
                        for nb in nbs[j]
                            w,f = navimap.edges[curr][nb]
                            navimap.edges[curr][nb] = (w, Floors[sflrs[j]])
                            navimap.edges[nb][curr] = (w, Floors[sflrs[j]])
                            curr = nb
                        end
                    end
                    prefx = string(prefx, sflrs[tid])
                elseif r1 == 3#wall
                    swlls = shuffle(walls)
                    for j=1:length(nbs)
                        curr = segments[i+1][2][1][1:2]
                        for nb in nbs[j]
                            w,f = navimap.edges[curr][nb]
                            navimap.edges[curr][nb] = j == tid ? (Walls[swlls[1]], f) : (Walls[swlls[rand(2:3)]], f)
                            navimap.edges[nb][curr] = j == tid ? (Walls[swlls[1]], f) : (Walls[swlls[rand(2:3)]], f)
                            curr = nb
                        end
                    end
                    prefx = string(prefx, swlls[1])
                end
            end
            ins = Instruction(name, split(prefx), path, mname, id)

            break
        end
    end
    return ins, navimap
end

combined_1011(name, id) = rand([turn_to_x_move_to_y, move_to_x_turn_to_y])(name, id)
combined_781011(name, id) = rand([turn_to_x_and_move, move_to_x_and_turn, turn_to_x_move_to_y, move_to_x_turn_to_y])(name, id)
combined_45781011(name, id) = rand([turn_and_move_to_x, lang_only, turn_to_x_and_move, move_to_x_and_turn, turn_to_x_move_to_y, move_to_x_turn_to_y])(name, id)
combined_1245781011(name, id) = rand([turn_to_x_move_to_y, move_to_x_turn_to_y, turn_and_move_to_x, lang_only, turn_to_x_and_move, move_to_x_and_turn, turn_to_x_move_to_y, move_to_x_turn_to_y])(name, id)

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
