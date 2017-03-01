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
            navimap = generate_navi_map(maze, ""; itemcountprobs=[0.05 0.3 0.65], iprob=0.15)
        end

        nodes, path = generate_path(maze, available)
        segments = segment_path(nodes)
        prefx = "turn to the "
        
        for i=1:length(segments)
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
                
                ins = Instruction(name, split(string(prefx, sits[1])), path, mname, id)
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

                    ins = Instruction(name, split(string(prefx, sflrs[tid], " hall")), path, mname, id)
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
                    
                    ins = Instruction(name, split(string(prefx, swlls[1])), path, mname, id)
                end
            end
            break
        end
    end
    return ins, navimap
end

"""
Available task functions:

turn_to_x
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
