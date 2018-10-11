using Logging

include("../SAILx/src/maze.jl")
include("flex.jl")
include("util.jl")
include("io.jl")

global models = nothing

global vocab = load("data/sailxdataset/vocab.jld", "vocab")
println("Vocab loaded")

function loadmodel(fname; flex=true)
    w = Dict()
    d = load(fname, "weights")
    for k in keys(d)
        if flex && startswith(k, "filter")
            w[k] = d[k]
        else
            w[k] = d[k]
        end
    end
    return w
end

models = [loadmodel("mbank/sailx_"*"$i"*".jld") for i=1:1]
println("Models loaded")

function random_map(hraw, wraw)
    h = parse(Int, hraw)
    w = parse(Int, wraw)
    maze, available = generate_maze(h, w; numdel=2)
    global navimap
    navimap = generate_navi_map(maze, "RANDOMMAZE"; itemcountprobs=[0.0 0.0 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.2 0.2], iprob=0.6)
end

function predict(instruction, initial)
    global models
    global vocab
    navimap = readmapsjson("navimap_demo.json")["demomap"]
    ins_text = split(instruction)
    words = ins_arr(vocab, ins_text)
    startpos = eval(parse(initial))
    args = Dict()
    args["hidden"] = 65
    args["limactions"] = 35
    args["bs"] = 1
    args["wvecs"] = false
    args["greedy"] = true
    args["percp"] = true
    args["preva"] = true
    args["worldatt"] = 100
    args["attinwatt"] = 0
    args["att"] = false
    args["inpout"] = true
    args["prevaout"] = false
    args["attout"] = false
    args["beamsize"] = 1
    args["encoding"] = "grid"
    args["atype"] = Array{Float32}
    
    predict_beam(models, words, navimap, startpos; args=args)
end

function demo()
    Logging.configure(level=DEBUG)
    Logging.configure(filename="demo.log")

    while true
        debug("\n***New instruction***")
        println("Enter the starting position: ")
        print("> ")
        initial = readline(STDIN)

        println("Enter the instruction: ")
        print("> ")
        instruction = readline(STDIN)
    
        actions = predict(instruction, initial)
        println("Actions: $actions")
    end
end

demo()
