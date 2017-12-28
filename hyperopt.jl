using ArgParse, DataFrames

include("io.jl")
include("gsection.jl")
include("util.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        ("--lr"; help = "learning rate"; default = 0.001; arg_type = Float64)
        ("--hidden"; help = "hidden size"; default = 120; arg_type = Int)
        ("--embed"; help = "embedding size"; default = 120; arg_type = Int)
        ("--epoch"; help = "number of epochs"; default = 50; arg_type = Int)
        ("--patience"; help = "patience param"; default = 10; arg_type = Int)
        ("--limactions"; arg_type = Int; default = 20)
        ("--window1"; help = "first dimension of the filters"; default = [1, 5, 1]; arg_type = Int; nargs = '+')
        ("--window2"; help = "second dimension of the filters"; default = [5, 5, 12]; arg_type = Int; nargs = '+')
        ("--filters"; help = "number of filters"; default = [40, 80, 50]; arg_type = Int; nargs = '+')
        ("--model"; help = "model file"; default = "flex.jl")
        ("--encdrops"; help = "dropout rates"; nargs = '+'; default = [0.0, 0.0]; arg_type = Float64)
        ("--decdrops"; help = "dropout rates"; nargs = '+'; default = [0.0, 0.0]; arg_type = Float64)
        ("--bs"; help = "batch size"; default = 1; arg_type = Int)
        ("--gclip"; help = "gradient clip"; default = 5.0; arg_type = Float64)
        ("--winit"; help = "scale the xavier"; default = 13.5; arg_type = Float64)
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
        ("--preva"; help = "use previous action"; action = :store_false)
        ("--att"; help = "use attention"; action = :store_true)
        ("--attinwatt"; help = "use attention"; arg_type = Int; default = 0)
        ("--worldatt"; help = "world attention"; arg_type = Int; default = 100)
        ("--inpout"; help = "direct connection from input to output"; action = :store_false)
        ("--prevaout"; help = "direct connection from prev action to output"; action = :store_true)
        ("--attout"; help = "direct connection from attention to output"; action = :store_true)
        ("--beamsize"; help = "world attention"; arg_type = Int; default = 10)
        ("--beam"; help = "activate beam search"; action = :store_true)
        ("--globalloss"; help = "use global loss"; action = :store_true)
        ("--level"; help = "log level"; default="info")
        ("--taskf"; help = "task folder"; default="")
        ("--limit"; help = "the error limit"; default=0.075; arg_type=Float64)
    end
    return parse_args(s)
end		

args = parse_commandline()

include(args["model"])

function pretrain(instances, maps, vocab, emb, args)
    w = nothing
    prms = nothing
    avg_lss = 0
    avg_acc = 0

    df = DataFrame(Batch=Int[], AvgLoss=Any[], AvgAcc=Any[], CLoss=Any[], CAcc=Any[])

    data, dat2 = nothing, nothing
    average_l = 0.0
    average_a = 0.0
    count = 0.0
    numins = 0
    for i=1:100:(length(instances)-200)
        numins= floor(Int, i / 100.0) + 1
        dat = instances[i:i+99]
        data = map(x -> build_instance(x, maps[x.map], vocab; encoding=args["encoding"], emb=nothing), dat)
        trn_data = minibatch(data;bs=args["bs"])
        
        dat2 = instances[(i+100):(i+199)]
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

            w = args["load"] != "" ? loadmodel(args["load"]; flex=true) : initweights(KnetArray{Float32}, args["hidden"], vocabsize, args["embed"], args["window"], world, args["filters"]; args=args, premb=premb, winit=args["winit"])

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
        @time tst_acc = test([w], tst_data, maps; args=args)
        
        average_l += lss * 100.0
        average_a += tst_acc * 100.0
        count += 100.0

        avg_lss = numins < 20 ? average_l / count : avg_lss*0.99 + 0.01*lss
        avg_acc = numins < 20 ? average_a / count : avg_acc*0.99 + 0.01*tst_acc

        info("Batch: $numins , Loss: $lss , Acc: $(tst_acc)")

        push!(df, (numins, avg_lss, avg_acc, lss, tst_acc))

        info("$(df[numins,:])")
        
        if (1.0 - avg_acc) < args["limit"]
            break
        end
    end

    info("***Weight Norms***")
    for k in keys(w)
        if startswith(k, "filter")
            for ind=1:length(w[k])
                info("$k , $ind : $(norm(vec(Array(w[k][ind])))))")
            end
        else
            info("$k : $(norm(Array(w[k]))) ")
        end
    end

    if !args["hopt"]
        writetable(args["savecsv"], df)
    end
    return (numins, w)
end

function hyperopt(instances, maps, vocab, emb, args)
    function xform_grid(x)
        winit,hidden,embl,f1,f2,f3,watt = exp.(x) .* [10, 80.0, 80.0, 40, 50, 80, 50]
        hidden = ceil(Int, hidden)
        embl = ceil(Int, embl)
        f1 = ceil(Int, f1)
        f2 = ceil(Int, f2)
        f3 = ceil(Int, f3)
        watt = floor(Int, watt)
        (winit,hidden,embl,f1,f2,f3,watt)
    end

    function xform_other(x)
        hidden,embl = exp.(x) .* [100.0, 100.0,]
        hidden = ceil(Int, hidden)
        embl = ceil(Int, embl)
        (hidden,embl)
    end

    function f(x)
        srand(args["seed"])
        if args["percp"] && args["encoding"] == "grid" && args["worldatt"] != 0
            winit, hidden, embl, f1, f2, f3, watt = xform_grid(x)
            args["winit"] = winit
            args["hidden"] = hidden
            args["embed"] = embl
            args["filters"] = [f1, f2, f3]
            args["worldatt"] = watt
            info("Config: ")
            info("winit: $winit , hidden: $hidden , embed: $embl , filters: $([f1, f2, f3]) , worldatt: $watt ")
        else
            hidden, embl = xform_other(x)
            args["hidden"] = hidden
            args["embed"] = embl
            info("Config: ")
            info("hidden: $hidden , embed: $embl ")
        end

        if args["hidden"] > 500 || args["embed"] > 500 || args["filters"][1] > 1500 || args["filters"][2] > 1500
            return NaN # prevent out of gpu
        end
        lss,_ = pretrain(instances, maps, vocab, emb, args)
        info("Config Loss: $lss")
        return lss
    end

    if args["percp"] && args["encoding"] == "grid" && args["worldatt"] != 0
        f0, x0 = goldensection(f, 7; verbose=true)
        xbest = xform_grid(x0)
   else
        f0, x0 = goldensection(f, 2; verbose=true)
        xbest = xform_other(x0)
    end
    return f0, xbest
end

function main()
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

    info("Reading instances")
    instances = readinsjson(args["taskf"] * "/instructions.json")
    info("Reading maps")
    maps = readmapsjson(args["taskf"] * "/maps.json")
    
    vocab = build_dict(instances)
    emb = args["wvecs"] ? load("data/embeddings.jld", "vectors") : nothing
    info("\nVocab: $(length(vocab))")

    f0, x0 = hyperopt(instances, maps, vocab, emb, args)
    info("Best loss: $f0")
    info("Best args: $x0")
end

main()