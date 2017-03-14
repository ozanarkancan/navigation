using ArgParse, DataFrames

include("io.jl")
include("datagen/data_generator.jl")
include("gsection.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        ("--lr"; help = "learning rate"; default = 0.001; arg_type = Float64)
        ("--hidden"; help = "hidden size"; default = 100; arg_type = Int)
        ("--embed"; help = "embedding size"; default = 100; arg_type = Int)
        ("--limactions"; arg_type = Int; default = 35)
        ("--window1"; help = "first dimension of the filters"; default = [29, 7, 5]; arg_type = Int; nargs = '+')
        ("--window2"; help = "second dimension of the filters"; default = [29, 7, 5]; arg_type = Int; nargs = '+')
        ("--filters"; help = "number of filters"; default = [200, 100, 50]; arg_type = Int; nargs = '+')
        ("--model"; help = "model file"; default = "flex.jl")
        ("--encdrops"; help = "dropout rates"; nargs = '+'; default = [0.0, 0.0]; arg_type = Float64)
        ("--decdrops"; help = "dropout rates"; nargs = '+'; default = [0.0, 0.0]; arg_type = Float64)
        ("--bs"; help = "batch size"; default = 1; arg_type = Int)
        ("--gclip"; help = "gradient clip"; default = 5.0; arg_type = Float64)
        ("--winit"; help = "scale the xavier"; default = 1.0; arg_type = Float64)
        ("--log"; help = "name of the log file"; default = "test.log")
        ("--save"; help = "model path"; default = "")
        ("--savecsv"; help = "csv path"; default = "")
        ("--load"; help = "model path"; default = "")
        ("--charenc"; help = "charecter embedding"; action = :store_true)
        ("--encoding"; help = "grid or multihot"; default = "grid")
        ("--wvecs"; help = "use word vectors"; action= :store_true)
        ("--greedy"; help = "deterministic or stochastic policy"; action = :store_false)
        ("--seed"; help = "seed number"; arg_type = Int; default = 123)
        ("--percp"; help = "use perception"; action = :store_false)
        ("--preva"; help = "use previous action"; action = :store_true)
        ("--att"; help = "use attention"; action = :store_true)
        ("--worldatt"; help = "world attention"; arg_type = Int; default = 0)
        ("--inpout"; help = "direct connection from input to output"; action = :store_false)
        ("--prevaout"; help = "direct connection from prev action to output"; action = :store_true)
        ("--attout"; help = "direct connection from attention to output"; action = :store_true)
        ("--level"; help = "log level"; default="info")
        ("--numbatch"; help = "number of batches"; default=100; arg_type=Int)
        ("--taskf"; help = "task function"; default="turn_to_x")
        ("--hopt"; help = "hyper parameter optimization"; action = :store_true)
    end
    return parse_args(s)
end		

args = parse_commandline()

include(args["model"])

function pretrain(vocab, emb, args)
    w = nothing
    prms = nothing
    avg_lss = 0
    avg_acc = 0

    df = DataFrame(Batch=Int[], AvgLoss=Any[], AvgAcc=Any[], CLoss=Any[], CAcc=Any[])
    taskf = eval(parse(args["taskf"]))

    data, dat2, maps, maps2 = (nothing, nothing, nothing, nothing)
    average = 0.0
    count = 0.0
    for i=1:args["numbatch"]
        dat, maps = dat2 == nothing ? generatedata(taskf) : (dat2, maps2)
        data = map(x -> build_instance(x, maps[x.map], vocab; encoding=args["encoding"], emb=nothing), dat)
        trn_data = minibatch(data;bs=args["bs"])
        
        dat2, maps2 = generatedata(taskf)
        tst_data = map(ins-> (ins, ins_arr(vocab, ins.text)), dat2)
        
        vdims = size(trn_data[1][2][1])

        vocabsize = length(vocab) + 1
        world = length(vdims) > 2 ? vdims[3] : vdims[2]

        if w == nothing
            embedsize = args["wvecs"] ? 300 : args["embed"]
            world = length(vdims) > 2 ? vdims[3] : vdims[2]

            premb = nothing
            if args["wvecs"]
                premb = zeros(Float32, vocabsize, embedsize)
                for k in keys(vocab)
                 premb[vocab[k], :] = emb[k]
                end
            end

            w = initweights(KnetArray, args["hidden"], vocabsize, args["embed"], args["window"], world, args["filters"]; args=args, premb=premb, winit=args["winit"])

            info("Model Prms:")
            for k in keys(w)
                info("$k : $(size(w[k])) ")
                if startswith(k, "filter")
                    for ind=1:length(w[k])
                        info("$k , $ind : $(size(w[k][ind]))")
                    end
                end
            end
        end

        if prms == nothing
            prms = initparams(w; args=args)
        end
        @time lss = train(w, prms, trn_data; args=args)
        @time tst_acc = test([w], tst_data, maps2; args=args)

        avg_lss = avg_lss == 0 ? lss : avg_lss*0.99 + 0.01*lss
        avg_acc = avg_acc == 0 ? tst_acc : avg_acc*0.99 + 0.01*tst_acc

        info("BatchNum: $i , Loss: $lss , Acc: $(tst_acc)")

        push!(df, (i, avg_lss, avg_acc, lss, tst_acc))

        info("$(df[i,:])")

        average += tst_acc * 100.0
        count += 100.0

        if (1.0 - avg_acc) < 1e-3
            break
        end
    end

    if !args["hopt"]
        writetable(args["savecsv"], df)
    end
    return 1 - (average / count)
end

function hyperopt(vocab, emb, args)
    function xform_grid(x)
        winit,hidden,embl,f1,f2,watt = exp(x) .* [5, 100.0, 100.0, 50, 5, 100]
        hidden = ceil(Int, hidden)
        embl = ceil(Int, embl)
        f1 = ceil(Int, f1)
        f2 = ceil(Int, f2)
        watt = floor(Int, watt)
        (winit,hidden,embl,f1,f2,watt)
    end
    
    function xform_other(x)
        hidden,embl = exp(x) .* [100.0, 100.0,]
        hidden = ceil(Int, hidden)
        embl = ceil(Int, embl)
        (hidden,embl)
    end

    function f(x)
        if args["percp"] && args["encoding"] == "grid"
            winit, hidden, embl, f1, f2, watt = xform_grid(x)
            args["winit"] = winit
            args["hidden"] = hidden
            args["embed"] = embl
            args["filters"] = [f1, f2]
            args["worldatt"] = att
        else
            hidden, embl = xform_other(x)
            args["hidden"] = hidden
            args["embed"] = embl
        end

        if args["hidden"] > 300 || args["embed"] > 500
            return NaN # prevent out of gpu
        end
        lss = pretrain(vocab, emb, args)
        info("Config Loss: $lss")
        return lss
    end

    if args["percp"] && args["encoding"] == "grid"
        f0, x0 = goldensection(f, 6)
        xbest = xform_grid(x0)
    else
        f0, x0 = goldensection(f, 2)
        xbest = xform_other(x0)
    end
    return f0, xbest
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

function mainpretraining()
    Logging.configure(filename=args["log"])
    if args["level"] == "info"
        Logging.configure(level=INFO)
    else
        Logging.configure(level=DEBUG)
    end

    args["window"] = collect(zip(args["window1"], args["window2"]))
    srand(args["seed"])
    info("*** Parameters ***")
    for k in keys(args); info("$k -> $(args[k])"); end

    grid, jelly, l = getallinstructions()
    lg = length(grid)
    lj = length(jelly)
    ll = length(l)
    dg = floor(Int, lg*0.5)
    dj = floor(Int, lj*0.5)
    dl = floor(Int, ll*0.5)

    maps = get_maps()

    vocab = build_dict(vcat(grid, jelly, l))
    emb = args["wvecs"] ? load("data/embeddings.jld", "vectors") : nothing
    info("\nVocab: $(length(vocab))")

    if args["hopt"]
        f0, x0 = hyperopt(vocab, emb, args)
        info("Best loss: $f0")
        info("Best args: $x0")
    else
        pretrain(vocab, emb, args)
    end
end

mainpretraining()
