using ArgParse

include("util.jl")
include("io.jl")
include("flex.jl")

function getconfig()
    args = Dict{String, Any}()
    
    #always same
    args["limactions"] = 20
    args["lr"] = 0.001
    args["trainfiles"] = ["grid_jelly.jld", "grid_l.jld", "l_jelly.jld"]
    args["testfiles"] = ["l", "jelly", "grid"]
    args["bs"] = 1
    args["greedy"] = true
    args["tunefor"] = "single"
    args["patience"] = 5
    args["vDev"] = true
    args["load"] = ""
    args["charenc"] = false
    args["encoding"] = "grid"

    #random
    args["hidden"] = rand(50:150)
    args["embed"] = args["hidden"]
    args["gclip"] = rand(1.0:5.0)
    args["encdrops"] = rand() < 0.2 ? 0.0 : rand()
    
    d1 = rand() < 0.2 ? 0.0 : rand()
    d2 = rand() < 0.2 ? 0.0 : rand()
    args["decdrops"] = [d1, d2]

    w1 = rand(3:2:35)
    w2 = rand(3:2:(39 - w1 + 1 - 2))
    w3 = rand(3:2:(39 - w1 - w2 + 2))

    args["window"] = [w1, w2, w3]
    args["filters"] = rand(20:300, 3)
    args["wvecs"] = true
    args["att"] = false
    args["percp"] = true
    args["preva"] = false
    args["inpout"] = true
    args["prevaout"] = false
    args["attout"] = false
    args["save"] = "mbank/hyperband.jld"
    return args
end

function execute(train_ins, dev_ins, maps, vocab, emb, args, res)
    data = map(x -> build_instance(x, maps[x.map], vocab; encoding=args["encoding"], emb=emb), vcat(train_ins[1], train_ins[2]))
    trn_data = minibatch(data;bs=args["bs"])

    train_data = map(ins-> (ins, (emb != nothing ? ins_arr_embed(emb, vocab, ins.text) : ins_arr(vocab, ins.text))), train_ins[1])
    vdims = size(trn_data[1][2][1])

    info("\nWorld: $(vdims)")

    vocabsize = args["wvecs"] ? 300 : length(vocab) + 1
    world = length(vdims) > 2 ? vdims[3] : vdims[2]

    w = initweights(KnetArray, args["hidden"], vocabsize, args["embed"], args["window"], world, args["filters"]; args=args)
    
    info("Model Prms:")
    for k in keys(w)
        info("$k : $(size(w[k])) ")
        if startswith(k, "filter")
            for i=1:length(w[k])
                info("$k , $i : $(size(w[k][i]))")
            end
        end
    end

    dev_data = map(ins -> (ins, (emb != nothing ? ins_arr_embed(emb, vocab, ins.text) : ins_arr(vocab, ins.text))), dev_ins)

    sofarbest = 0.0

    prms_sp = initparams(w; args=args)
    patience = 0
    ld = length(train_ins)
    totalsource = floor(Int, ld*8.0*res)
    ep = 1
    while true
        shuffle!(trn_data)
        @time lss = train(w, prms_sp, trn_data; args=args, updatelimit=totalsource)

        dev_acc = 0

        @time dev_acc = test([w], dev_data, maps; args=args)

        if dev_acc > sofarbest
            sofarbest = dev_acc
            patience = 0
        else
            patience += 1
        end

        info("Epoch: $(ep), trn loss: $(lss) , single acc: $(dev_acc) , $(dev_ins[1].map)")
        ep += 1

        if patience >= args["patience"]
            break
        end

        totalsource -= ld
        if totalsource <= 0
            break
        end
    end

    return sofarbest
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


function hyperband(getconfig, getloss, maxresource, reduction=3)
    @show smax = floor(Int, log(maxresource)/log(reduction))
    @show B = (smax + 1) * maxresource
    best = (Inf,)
    for s in smax:-1:0
        n = ceil(Int, (B/maxresource)*((reduction^s)/(s+1)))
        r = maxresource / (reduction^s)
        curr = halving(getconfig, getloss, n, r, reduction, s)
        if curr[1] < best[1]; (best=curr); end
    end
    return best
end

function halving(getconfig, getloss, n, r=1, reduction=3, s=round(Int, log(n)/log(reduction)))
    best = (Inf,)
    T = [ getconfig() for i=1:n ]
    for i in 0:s
        ni = floor(Int,n/(reduction^i))
        ri = r*(reduction^i)
        println((:s,s,:n,n,:r,r,:i,i,:ni,ni,:ri,ri,:T,length(T)))
        L = [ getloss(t, ri) for t in T ]
        l = sortperm(L); l1=l[1]
        L[l1] < best[1] && (best = (L[l1],ri,T[l1]); printbest(best))
        T = T[l[1:floor(Int,ni/reduction)]]
    end
    printbest(best)
    return best
end

function printbest(best)
    info("\n*** Best: ***")
    info("Loss: $(best[1])")
    info("Resource: $(best[2])")
    info("Args")
    for k in keys(best[3]); info("k: $k , val: $(best[3][k])"); end
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        ("--log"; help = "name of the log file"; default = "test.log")
        ("--seed"; help = "seed number"; arg_type = Int; default = 123)
    end
    return parse_args(s)
end

function mainhyperband()
    args = parse_commandline()

	Logging.configure(filename=args["log"])
	Logging.configure(level=INFO)
	
	grid, jelly, l = getallinstructions()
	lg = length(grid)
	lj = length(jelly)
	ll = length(l)
	dg = floor(Int, lg*0.5)
	dj = floor(Int, lj*0.5)
	dl = floor(Int, ll*0.5)

    trn = (jelly, l)
    dev1 = grid[(1+dg):end]
    dev2 = grid[1:dg]
    
    maps = get_maps()

	vocab = build_dict(vcat(grid, jelly, l))

    #1 resource = 0.25 of data
    function getloss(args, res)
        info("Args")
        for k in keys(args); info("k: $k , val: $(args[k])"); end;

        info("Res: $res")
        emb = args["wvecs"] ? load("data/embeddings.jld", "vectors") : nothing
        lss = 100
        try
            acc1 = execute(trn, dev1, maps, vocab, emb, args, res)
            acc2 = execute(trn, dev2, maps, vocab, emb, args, res)
            lss = 1 - (acc1 + acc2) / 2
        catch
            info("Model has been failed")
            lss = 1000
        end
        info("Loss: $lss")

        return lss
    end

    maxresource = 81
    reduction = 3
    best = hyperband(getconfig, getloss, maxresource, reduction)
    info("### Final ###")
    printbest(best)
end

mainhyperband()
