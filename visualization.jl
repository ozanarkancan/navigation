using ArgParse, CSV, DataFrames

include("util.jl")
include("io.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        ("--hidden"; help = "hidden size"; default = 120; arg_type = Int)
        ("--embed"; help = "embedding size"; default = 120; arg_type = Int)
        ("--limactions"; arg_type = Int; default = 35)
        ("--trainfiles"; help = "built training jld file"; default = ["grid_jelly.jld", "grid_l.jld", "l_jelly.jld"]; nargs = '+')
        ("--testfiles"; help = "test file as regular instruction file(json)"; default = ["l", "jelly", "grid"]; nargs = '+')
        ("--filters"; help = "number of filters"; default = [300, 150, 50]; arg_type = Int; nargs = '+')
        ("--model"; help = "model file"; default = "flex.jl")
        ("--encdrops"; help = "dropout rates"; nargs = '+'; default = [0.5, 0.5]; arg_type = Float64)
        ("--decdrops"; help = "dropout rates"; nargs = '+'; default = [0.5, 0.5]; arg_type = Float64)
        ("--bs"; help = "batch size"; default = 1; arg_type = Int)
        ("--gclip"; help = "gradient clip"; default = 5.0; arg_type = Float64)
        ("--winit"; help = "scale the xavier"; default = 1.0; arg_type = Float64)
        ("--log"; help = "name of the log file"; default = "test.log")
        ("--save"; help = "model path"; default = "")
        ("--load"; help = "model path"; default = []; nargs='+')
        ("--vDev"; help = "vDev or vTest"; action = :store_false)
        ("--charenc"; help = "charecter embedding"; action = :store_true)
        ("--encoding"; help = "grid or multihot"; default = "grid")
        ("--wvecs"; help = "use word vectors"; action= :store_true)
        ("--greedy"; help = "deterministic or stochastic policy"; action = :store_false)
        ("--seed"; help = "seed number"; arg_type = Int; default = 123)
        ("--test"; help = "0,1,2,3,4,5,6 (all, l[1], l[2], jelly[1], jelly[2], grid[1], grid[2])"; arg_type = Int)
        ("--percp"; help = "use perception"; action = :store_false)
        ("--preva"; help = "use previous action"; action = :store_false)
        ("--worldatt"; help = "world attention"; arg_type = Int; default = 100)
        ("--attinwatt"; help = "use attention"; arg_type = Int; default = 0)
        ("--att"; help = "use attention"; action = :store_true)
        ("--inpout"; help = "direct connection from input to output"; action = :store_false)
        ("--prevaout"; help = "direct connection from prev action to output"; action = :store_true)
        ("--attout"; help = "direct connection from attention to output"; action = :store_true)
        ("--level"; help = "log level"; default="info")
        ("--beamsize"; help = "world attention"; arg_type = Int; default = 10)
        ("--beam"; help = "activate beam search"; action = :store_true)
        ("--categorical"; help = "dump categorical"; default="")
        ("--sailx"; help = "folder for sailx"; default="")
        ("--oldvdev"; help = "vDev training scheme"; default = [0]; nargs='+'; arg_type =Int)
        ("--oldvtest"; help = "vTest training scheme"; default = [0]; nargs='+'; arg_type =Int)

    end
    return parse_args(s)
end		

args = parse_commandline()

include(args["model"])

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

function sailx(args)
    info("Reading data")
    devins = readinsjson(string(args["sailx"], "/dev/instructions.json"))
    testins = readinsjson(string(args["sailx"], "/test/instructions.json"))
    maps = readmapsjson(string(args["sailx"], (args["vDev"] ? "/dev/maps.json" : "/test/maps.json")))

    vocab = nothing
    if isfile(string(args["sailx"], "/vocab.jld"))
        vocab = load(string(args["sailx"], "/vocab.jld"), "vocab")
    else
        info("Vocab generation")
        trainins = readinsjson(string(args["sailx"], "/train/instructions.json"))
        vocab = build_dict(vcat(trainins, devins, testins))
        save(string(args["sailx"], "/vocab.jld"), "vocab", vocab) 
    end

    models = Any[]
    for mfile in args["load"]
        push!(models, loadmodel(mfile; flex=true))
        w = models[end]
        info("Model Prms:")
        for k in keys(w)
            info("$k : $(size(w[k])) ")
            if startswith(k, "filter")
                for i=1:length(w[k])
                    info("$k , $i : $(size(w[k][i]))")
                end
            end
        end
    end

    test_data = map(ins-> (ins, ins_arr(vocab, ins.text)), args["vDev"] ? devins : testins)
    
    tasks = Dict()
    df = DataFrame(task=String[], text=String[], correct=Float32[], firstatt=Array{Array{Float32}, 1}(), lastatt=Array{Array{Float32}, 1}())
    for ind=1:length(test_data)
        datum = test_data[ind]
        fname = join(split(datum[1].fname,"_")[1:(end-1)], "_")
        if !(fname in collect(keys(tasks)))
            tasks[fname] = [0.0, 0.0, 0.0]#beam=1, beam=10, count
        end

        tst_acc, att_ps  = test(models, [datum], maps; args=args)
        tst_acc_beam = test_beam(models, [datum], maps; args=args)
        tasks[fname][1] = tasks[fname][1] + tst_acc
        tasks[fname][2] = tasks[fname][2] + tst_acc_beam
        tasks[fname][3] = tasks[fname][3] + 1

        push!(df, (fname, join(testins[ind].text, " "), tst_acc, att_ps[1][1], att_ps[1][end]))
    end
    
    singles = 0.0
    beams = 0.0
    for (k,v) in tasks
        singles += v[1]
        beams += v[2]
        info("Task: $k Single: $(v[1] / v[3]) , Beam Single: $(v[2] / v[3])")
    end
    l = length(test_data)
    info("Total: Single: $(singles / l) , Beam: $(beams / l)")
    CSV.write("attentions.csv", df)
end

function mainflex()
    Logging.configure(filename=args["log"])
    if args["level"] == "info"
        Logging.configure(level=INFO)
    else
        Logging.configure(level=DEBUG)
    end
    srand(args["seed"])
    info("*** Parameters ***")
    for k in keys(args); info("$k -> $(args[k])"); end
    sailx(args)
end

mainflex()
