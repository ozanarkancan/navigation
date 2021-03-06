using ArgParse

include("util.jl")
include("io.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        ("--lr"; help = "learning rate"; default = 0.001; arg_type = Float64)
        ("--hidden"; help = "hidden size"; default = 120; arg_type = Int)
        ("--embed"; help = "embedding size"; default = 120; arg_type = Int)
        ("--limactions"; arg_type = Int; default = 35)
        ("--epoch"; help = "number of epochs"; default = 2; arg_type = Int)
        ("--trainfiles"; help = "built training jld file"; default = ["grid_jelly.jld", "grid_l.jld", "l_jelly.jld"]; nargs = '+')
        ("--testfiles"; help = "test file as regular instruction file(json)"; default = ["l", "jelly", "grid"]; nargs = '+')
        ("--window1"; help = "first dimension of the filters"; default = [1, 5, 1]; arg_type = Int; nargs = '+')
        ("--window2"; help = "second dimension of the filters"; default = [5, 5, 12]; arg_type = Int; nargs = '+')
        ("--filters"; help = "number of filters"; default = [40, 80, 50]; arg_type = Int; nargs = '+')
        ("--model"; help = "model file"; default = "flex.jl")
        ("--encdrops"; help = "dropout rates"; nargs = '+'; default = [0.6]; arg_type = Float64)
        ("--decdrops"; help = "dropout rates"; nargs = '+'; default = [0.5, 0.8]; arg_type = Float64)
        ("--bs"; help = "batch size"; default = 1; arg_type = Int)
        ("--gclip"; help = "gradient clip"; default = 5.0; arg_type = Float64)
        ("--winit"; help = "scale the xavier"; default = 13.5; arg_type = Float64)
        ("--log"; help = "name of the log file"; default = "test.log")
        ("--save"; help = "model path"; default = "")
        ("--patience"; help = "patience param"; default = 10; arg_type = Int)
        ("--tunefor"; help = "tune for (single or paragraph)"; default = "single")
        ("--load"; help = "model path"; default = "")
        ("--pretrain"; help = "model path"; default = "")
        ("--vDev"; help = "vDev or vTest"; action = :store_false)
        ("--oldvdev"; help = "vDev training scheme"; default = [0]; nargs='+'; arg_type =Int)
        ("--oldvtest"; help = "vTest training scheme"; default = [0]; nargs='+'; arg_type =Int)
        ("--charenc"; help = "charecter embedding"; action = :store_true)
        ("--encoding"; help = "grid or multihot"; default = "grid")
        ("--wvecs"; help = "use word vectors"; action= :store_true)
        ("--greedy"; help = "deterministic or stochastic policy"; action = :store_false)
        ("--seed"; help = "seed number"; arg_type = Int; default = 1234)
        ("--percp"; help = "use perception"; action = :store_false)
        ("--preva"; help = "use previous action"; action = :store_false)
        ("--worldatt"; help = "world attention"; arg_type = Int; default = 100)
        ("--att"; help = "use attention"; action = :store_true)
        ("--attinwatt"; help = "use attention"; arg_type = Int; default = 0)
        ("--inpout"; help = "direct connection from input to output"; action = :store_false)
        ("--prevaout"; help = "direct connection from prev action to output"; action = :store_true)
        ("--attout"; help = "direct connection from attention to output"; action = :store_true)
        ("--level"; help = "log level"; default="info")
        ("--beamsize"; help = "world attention"; arg_type = Int; default = 10)
        ("--beam"; help = "activate beam search"; action = :store_false)
        ("--globalloss"; help = "use global loss"; action = :store_true)
        ("--sailx"; help = "folder of sailx data"; default = "")
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

    ptr = args["pretrain"]
    w = ptr != "" ? loadmodel(ptr; flex=true) : initweights(KnetArray{Float32}, args["hidden"], vocabsize, args["embed"], args["window"], world, args["filters"]; args=args, premb=premb, winit=args["winit"])

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
    dev_d = data != nothing ? minibatch(data;bs=args["bs"]) : nothing

    test_data = map(ins-> (ins, ins_arr(vocab, ins.text)), test_ins)
    test_data_grp = map(x->map(ins-> (ins, ins_arr(vocab, ins.text)),x), group_singles(test_ins))

    globalbest = 0.0

    prms_sp = initparams(w; args=args)
    patience = 0
    sofarbest = 0.0
    for i=1:args["epoch"]
        shuffle!(trn_data)
        @time lss = !args["globalloss"] ? train(w, prms_sp, trn_data; args=args) : train_global(w, prms_sp, train_data, maps; args=args)
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

function execute_sailx(train_ins, test_ins, maps, vocab, emb, args; dev_ins=nothing)
    trn_data = map(x -> build_instance(x, maps[x.map], vocab; encoding=args["encoding"], emb=nothing), train_ins)
    trn_data = minibatch(trn_data;bs=args["bs"])

    train_data = map(ins-> (ins, ins_arr(vocab, ins.text)), train_ins)
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

    ptr = args["pretrain"]
    w = ptr != "" ? loadmodel(ptr; flex=true) : initweights(KnetArray{Float32}, args["hidden"], vocabsize, args["embed"], args["window"], world, args["filters"]; args=args, premb=premb, winit=args["winit"])

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
    data = dev_ins != nothing ? map(x -> build_instance(x, maps[x.map], vocab; encoding=args["encoding"], emb=nothing), dev_ins) : nothing
    dev_d = data != nothing ? minibatch(data;bs=args["bs"]) : nothing

    test_data = map(ins-> (ins, ins_arr(vocab, ins.text)), test_ins)

    globalbest = 0.0

    prms_sp = initparams(w; args=args)
    patience = 0
    sofarbest = 0.0
    for i=1:args["epoch"]
        shuffle!(trn_data)
        @time lss = !args["globalloss"] ? train(w, prms_sp, trn_data; args=args) : train_global(w, prms_sp, train_data, maps; args=args)
        @time train_acc = test([w], train_data, maps; args=args)
        @time tst_acc = test([w], test_data, maps; args=args)
        @time trnloss = train_loss(w, trn_data; args=args)
        
        if args["beam"]
            @time tst_acc_beam = test_beam([w], test_data, maps; args=args)
        end

        dev_acc = 0
        dev_prg_acc = 0
        dev_loss = 0

        tunefordev = 0
        if args["vDev"]
            @time dev_acc = test([w], dev_data, maps; args=args)
            @time dev_loss = train_loss(w, dev_d; args=args)
            if args["beam"]
                @time dev_acc_beam = test_beam([w], dev_data, maps; args=args)
            end
            tunefordev = args["tunefor"] == "single" ? dev_acc : dev_prg_acc
        end

        tunefor = args["tunefor"] == "single" ? tst_acc : tst_prg_acc
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
            info("Epoch: $(i), trn loss: $(lss) , single dev acc: $(dev_acc)")
            if args["beam"]
                info("Beam: $(i), single dev: $(dev_acc_beam) , single tst: $(tst_acc_beam)")
            end
            info("TestIt: $(i), trn loss: $(lss) , single tst acc: $(tst_acc)")
            info("Losses: $(i), trn loss: $(trnloss) , dev loss: $(dev_loss)")
        else
            info("Epoch: $(i), trn loss: $(lss) , single tst acc: $(tst_acc)")
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

function sail(args)
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

function sail_vdev(args)
    grid, jelly, l = getallinstructions()
    lg = length(grid)
    lj = length(jelly)
    ll = length(l)
    dg = floor(Int, lg*0.1)
    dj = floor(Int, lj*0.1)
    dl = floor(Int, ll*0.1)

    #trainfiles
    #1: grid+jelly
    #2: grid+l
    #3: jelly+l
    
    trainins = [(grid, jelly, dg, dj), (grid, l, dg, dl), (jelly, l, dj, dl)]
    testins = [l, jelly, grid]

    maps = get_maps()

    vocab = build_dict(vcat(grid, jelly, l))
    emb = args["wvecs"] ? load("data/embeddings.jld", "vectors") : nothing
    info("\nVocab: $(length(vocab))")

    base_s = args["save"]
    base_l = args["load"]
    for i in args["oldvdev"]
        for j=1:10
            args["save"] = string(base_s, "_", j, "_", args["trainfiles"][i])
            if base_l != ""
                args["load"] = string(base_l, "_", j, "_", args["trainfiles"][i])
            end
            ttuple = trainins[i]
            t = (vcat(j == 1 ? [] : ttuple[1][1:(j-1)*ttuple[3]],
                      j == 10 ? [] : ttuple[1][j*ttuple[3]+1:end]),
                 vcat(j == 1 ? [] : ttuple[2][1:(j-1)*ttuple[4]],
                      j == 10 ? [] : ttuple[2][j*ttuple[4]+1:end]))

            d = vcat(ttuple[1][(j-1)*ttuple[3]+1:j*ttuple[3]],
                     ttuple[2][(j-1)*ttuple[4]+1:j*ttuple[4]])
            info("i: $i j: $j lt: $(length(t)) ld: $(length(d))")

            execute(t, testins[i], maps, vocab, emb, args; dev_ins=d)
        end
    end
end

function sail_vtest(args)
    grid, jelly, l = getallinstructions()
    lg = length(grid)
    lj = length(jelly)
    ll = length(l)

    trainins = [(grid, jelly), (grid, l), (jelly, l)]
    testins = [l, jelly, grid]
    maps = get_maps()

    vocab = build_dict(vcat(grid, jelly, l))
    emb = args["wvecs"] ? load("data/embeddings.jld", "vectors") : nothing
    info("\nVocab: $(length(vocab))")

    base_s = args["save"]
    base_l = args["load"]
    for i in args["oldvtest"]
        args["save"] = string(base_s, "_vtest_", args["trainfiles"][i])
        if base_l != ""
            args["load"] = string(base_l, "_vtest_", args["trainfiles"][i])
        end
        execute(trainins[i], testins[i], maps, vocab, emb, args)
    end
end

function sailx(args)
    info("Reading training data")
    trainins = readinsjson(string(args["sailx"], "/train/instructions.json"))
    info("Reading dev data")
    devins = readinsjson(string(args["sailx"], "/dev/instructions.json"))
    info("Reading test data")
    testins = readinsjson(string(args["sailx"], "/test/instructions.json"))

    info("Reading maps")
    maps = readmapsjson(string(args["sailx"], "/train/maps.json"))
    merge!(maps, readmapsjson(string(args["sailx"], "/dev/maps.json")))
    vocab = build_dict(vcat(trainins, devins, testins))

    emb = args["wvecs"] ? load("data/embeddings.jld", "vectors") : nothing
    info("\nVocab: $(length(vocab))")
    
    execute_sailx(trainins, devins, maps, vocab, emb, args)
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

    if args["sailx"] == ""
        if args["oldvtest"][1] != 0
            sail_vtest(args)
        elseif args["oldvdev"][1] != 0
            sail_vdev(args)
        else
            sail(args)
        end
    else
        sailx(args)
    end
end

mainflex()
