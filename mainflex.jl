using ArgParse

include("util.jl")
include("io.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        ("--lr"; help = "learning rate"; default = 0.001; arg_type = Float64)
        ("--hidden"; help = "hidden size"; default = 50; arg_type = Int)
        ("--embed"; help = "embedding size"; default = 50; arg_type = Int)
        ("--limactions"; arg_type = Int; default = 35)
        ("--epoch"; help = "number of epochs"; default = 2; arg_type = Int)
        ("--trainfiles"; help = "built training jld file"; default = ["grid_jelly.jld", "grid_l.jld", "l_jelly.jld"]; nargs = '+')
        ("--testfiles"; help = "test file as regular instruction file(json)"; default = ["l", "jelly", "grid"]; nargs = '+')
        ("--window1"; help = "first dimension of the filters"; default = [29, 7, 5]; arg_type = Int; nargs = '+')
        ("--window2"; help = "second dimension of the filters"; default = [29, 7, 5]; arg_type = Int; nargs = '+')
        ("--filters"; help = "number of filters"; default = [300, 150, 50]; arg_type = Int; nargs = '+')
        ("--model"; help = "model file"; default = "flex.jl")
        ("--encdrops"; help = "dropout rates"; nargs = '+'; default = [0.5, 0.5]; arg_type = Float64)
        ("--decdrops"; help = "dropout rates"; nargs = '+'; default = [0.5, 0.5]; arg_type = Float64)
        ("--bs"; help = "batch size"; default = 1; arg_type = Int)
        ("--gclip"; help = "gradient clip"; default = 5.0; arg_type = Float64)
        ("--winit"; help = "scale the xavier"; default = 1.0; arg_type = Float64)
        ("--log"; help = "name of the log file"; default = "test.log")
        ("--save"; help = "model path"; default = "")
        ("--patience"; help = "patience param"; default = 10; arg_type = Int)
        ("--tunefor"; help = "tune for (single or paragraph)"; default = "single")
        ("--load"; help = "model path"; default = "")
        ("--vDev"; help = "vDev or vTest"; action = :store_false)
        ("--charenc"; help = "charecter embedding"; action = :store_true)
        ("--encoding"; help = "grid or multihot"; default = "grid")
        ("--wvecs"; help = "use word vectors"; action= :store_true)
        ("--greedy"; help = "deterministic or stochastic policy"; action = :store_false)
        ("--seed"; help = "seed number"; arg_type = Int; default = 123)
        ("--percp"; help = "use perception"; action = :store_false)
        ("--preva"; help = "use previous action"; action = :store_true)
        ("--worldatt"; help = "world attention"; arg_type = Int; default = 0)
        ("--att"; help = "use attention"; action = :store_true)
        ("--inpout"; help = "direct connection from input to output"; action = :store_true)
        ("--prevaout"; help = "direct connection from prev action to output"; action = :store_true)
        ("--attout"; help = "direct connection from attention to output"; action = :store_true)
        ("--level"; help = "log level"; default="info")
        ("--beamsize"; help = "world attention"; arg_type = Int; default = 10)
        ("--beam"; help = "activate beam search"; action = :store_true)
    end
    return parse_args(s)
end		

args = parse_commandline()

include(args["model"])

function execute(train_ins, test_ins, maps, vocab, emb, args; dev_ins=nothing)
    data = map(x -> build_instance(x, maps[x.map], vocab; encoding=args["encoding"], emb=nothing), vcat(train_ins[1], train_ins[2]))
    trn_data = minibatch(data;bs=args["bs"])

    train_data = map(ins-> (ins, ins_arr(vocab, ins.text)), vcat(train_ins[1], train_ins[2]))
    vdims = size(trn_data[1][2][1])

    info("\nWorld: $(vdims)")

    embedsize = args["wvecs"] ? 300 : args["embed"]
    vocabsize = length(vocab) + 1
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
            for i=1:length(w[k])
                info("$k , $i : $(size(w[k][i]))")
            end
        end
    end

    dev_data = dev_ins != nothing ? map(ins -> (ins, ins_arr(vocab, ins.text)), dev_ins) : nothing
    dev_data_grp = dev_ins != nothing ? map(x->map(ins->(ins, ins_arr(vocab, ins.text)),x), group_singles(dev_ins)) : nothing
    data = dev_ins != nothing ? map(x -> build_instance(x, maps[x.map], vocab; encoding=args["encoding"], emb=nothing), dev_ins) : nothing
    dev_d = minibatch(data;bs=args["bs"])

    test_data = map(ins-> (ins, ins_arr(vocab, ins.text)), test_ins)
    test_data_grp = map(x->map(ins-> (ins, ins_arr(vocab, ins.text)),x), group_singles(test_ins))

    globalbest = 0.0

    prms_sp = initparams(w; args=args)
    patience = 0
    sofarbest = 0.0
    for i=1:args["epoch"]
        shuffle!(trn_data)
        @time lss = train(w, prms_sp, trn_data; args=args)
        @time train_acc = test([w], train_data, maps; args=args)
        @time tst_acc = test([w], test_data, maps; args=args)
        @time tst_prg_acc = test_paragraph([w], test_data_grp, maps; args=args)
        @time trnloss = train_loss(w, trn_data; args=args)
        
        if args["beam"]
            @time tst_acc_beam = test_beam([w], test_data, maps; args=args)
            @time tst_prg_acc_beam = test_paragraph_beam([w], test_data_grp, maps; args=args)
        end

        dev_acc = 0
        dev_prg_acc = 0
        dev_loss = 0

        if args["vDev"]
            @time dev_acc = test([w], dev_data, maps; args=args)
            @time dev_prg_acc = test_paragraph([w], dev_data_grp, maps; args=args)
            @time dev_loss = train_loss(w, dev_d; args=args)
            if args["beam"]
                @time dev_acc_beam = test_beam([w], dev_data, maps; args=args)
                @time dev_prg_acc_beam = test_paragraph_beam([w], dev_data_grp, maps; args=args)
            end
        end

        tunefor = args["tunefor"] == "single" ? tst_acc : tst_prg_acc
        tunefordev = args["tunefor"] == "single" ? dev_acc : dev_prg_acc
        tunefor = args["vDev"] ? tunefordev : tunefor

        if tunefor > sofarbest
            sofarbest = tunefor
            patience = 0
            if sofarbest > globalbest
                globalbest = sofarbest
                info("Saving the model...")
                savemodel(w, args["save"]; flex=true)
            end
        else
            patience += 1
        end

        if args["vDev"]
            info("Epoch: $(i), trn loss: $(lss) , single dev acc: $(dev_acc) , paragraph acc: $(dev_prg_acc) , $(dev_ins[1].map)")
            if args["beam"]
                info("Beam: $(i), single dev: $(dev_acc_beam) , paragraph acc: $(dev_prg_acc_beam) , single tst: $(tst_acc_beam) , paragraph tst: $(tst_prg_acc_beam) $(dev_ins[1].map)")
            end
            info("TestIt: $(i), trn loss: $(lss) , single tst acc: $(tst_acc) , paragraph acc: $(tst_prg_acc) , $(test_ins[1].map)")
            info("Losses: $(i), trn loss: $(trnloss) , dev loss: $(dev_loss) , $(dev_ins[1].map)")
        else
            info("Epoch: $(i), trn loss: $(lss) , single tst acc: $(tst_acc) , paragraph acc: $(tst_prg_acc) , $(test_ins[1].map)")
        end
        info("Train: $(i) , trn acc: $(train_acc)")

        if patience >= args["patience"]
            break
        end
    end
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

function mainflex()
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

    trainins = [(grid, jelly), (grid, l), (jelly, l)]
    devins = [l[1:dl], l[(dl+1):end], jelly[1:dj], jelly[(dj+1):end], grid[1:dg], grid[(dg+1):end]]
    testins = [l[(dl+1):end], l[1:dl], jelly[(dj+1):end], jelly[1:dj], grid[(dg+1):end], grid[1:dg]]
    maps = get_maps()

    vocab = build_dict(vcat(grid, jelly, l))
    emb = args["wvecs"] ? load("data/embeddings.jld", "vectors") : nothing
    info("\nVocab: $(length(vocab))")

    base_s = args["save"]
    base_l = args["load"]
    for i in [3,2,1]
        for j=1:2
            args["save"] = string(base_s, "_", j, "_", args["trainfiles"][i])
            if base_l != ""
                args["load"] = string(base_l, "_", j, "_", args["trainfiles"][i])
            end
            execute(trainins[i], testins[(i-1)*2+j], maps, vocab, emb, args; dev_ins=devins[(i-1)*2+j])
        end
    end
end

mainflex()
