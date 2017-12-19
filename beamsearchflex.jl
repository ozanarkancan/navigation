using ArgParse, DataFrames

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

function sail(args)
    grid, jelly, l = getallinstructions()
    lg = length(grid)
    lj = length(jelly)
    ll = length(l)
    dg = floor(Int, lg*0.5)
    dj = floor(Int, lj*0.5)
    dl = floor(Int, ll*0.5)

    testins = [l[(dl+1):end], l[1:dl], jelly[(dj+1):end], jelly[1:dj], grid[(dg+1):end], grid[1:dg]]
    maps = get_maps()

    vocab = build_dict(vcat(grid, jelly, l))
    emb = args["wvecs"] ? load("data/embeddings.jld", "vectors") : nothing
    info("\nVocab: $(length(vocab))")
    
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

    test_ins = args["test"] != 0 ? testins[args["test"]] : vcat(grid, jelly, l)
    test_data = map(ins-> (ins, ins_arr(vocab, ins.text)), test_ins)
    test_data_grp = map(x->map(ins-> (ins, ins_arr(vocab, ins.text)),x), group_singles(test_ins))
    
    if args["categorical"] != ""
        df = DataFrame(fname=Any[], text=Any[], actions=Any[], Accuracy=Float64[], id=Any[])
        info("Model Prms:")
        w = models[1]
        for k in keys(w)
            info("$k : $(size(w[k])) ")
            if startswith(k, "filter")
                for i=1:length(w[k])
                    info("$k , $i : $(size(w[k][i]))")
                end
            end
        end

        for d in test_data
            acc = test_beam(models, [d], maps; args=args)
            push!(df, (d[1].fname, join(d[1].text, " "), getactions(d[1].path), acc, d[1].id))
        end

        writetable(args["categorical"], df)
    else

        @time tst_acc = test(models, test_data, maps; args=args)
        @time tst_prg_acc = test_paragraph(models, test_data_grp, maps; args=args)
        @time tst_acc_beam = test_beam(models, test_data, maps; args=args)
        @time tst_prg_acc_beam = test_paragraph_beam(models, test_data_grp, maps; args=args)

        info("Single: $tst_acc , Paragraph: $tst_prg_acc , Beam Single: $tst_acc_beam , Beam Paragraph: $tst_prg_acc_beam")
    end
    
end

function sailx(args)
    trainins = readinsjson(string(args["sailx"], "/train/instructions.json"))
    devins = readinsjson(string(args["sailx"], "/dev/instructions.json"))
    testins = readinsjson(string(args["sailx"], "/test/instructions.json"))
    maps = readmapsjson(string(args["sailx"], "/test/maps.json"))
    vocab = build_dict(vcat(trainins, devins, testins))

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

    test_data = map(ins-> (ins, ins_arr(vocab, ins.text)), testins)
    
    if args["categorical"] != ""
        df = DataFrame(fname=Any[], text=Any[], actions=Any[], Accuracy=Float64[], id=Any[])
        info("Model Prms:")
        w = models[1]
        for k in keys(w)
            info("$k : $(size(w[k])) ")
            if startswith(k, "filter")
                for i=1:length(w[k])
                    info("$k , $i : $(size(w[k][i]))")
                end
            end
        end

        for d in test_data
            acc = test_beam(models, [d], maps; args=args)
            push!(df, (d[1].fname, join(d[1].text, " "), getactions(d[1].path), acc, d[1].id))
        end

        writetable(args["categorical"], df)
    else

        @time tst_acc = test(models, test_data, maps; args=args)
        @time tst_acc_beam = test_beam(models, test_data, maps; args=args)

        info("Single: $tst_acc , Beam Single: $tst_acc_beam")
    end
    
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

    if args["sailx"] == ""
        sail(args)
    else
        sailx(args)
    end

end

mainflex()
