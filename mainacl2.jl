using ArgParse

include("util.jl")
include("io.jl")

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin
		"--lr"
			help = "learning rate"
			default = 0.001
			arg_type = Float64
		"--hidden"
			help = "hidden size"
			default = 128
			arg_type = Int
		"--embed"
			help = "embedding size"
			default = 128
			arg_type = Int
		"--limactions"
			arg_type = Int
			default = 35
		"--epoch"
			help = "number of epochs"
			default = 100
			arg_type = Int
		"--trainfiles"
			help = "built training jld file"
			default = ["grid_jelly.jld", "grid_l.jld", "l_jelly.jld"]
			nargs = '+'
		"--testfiles"
			help = "test file as regular instruction file(json)"
			default = ["l",
				"jelly",
				"grid"]
			nargs = '+'
		"--window"
			help = "size of the filter"
			default = [3]
			arg_type = Int
			nargs = '+'
		"--filters"
			help = "number of filters"
			default = [30]
			arg_type = Int
			nargs = '+'
		"--model"
			help = "model file"
			default = "model01.jl"
		"--pdrops"
			help = "dropout rates"
			nargs = '+'
			default = [0.2, 0.5, 0.5]
			arg_type = Float64
		"--pdrops_dec"
			help = "dropout rates"
			nargs = '+'
			default = [0.2, 0.5, 0.5]
			arg_type = Float64
		"--bs"
			help = "batch size"
			default = 100
			arg_type = Int
		"--gclip"
			help = "gradient clip"
			default = 10.0
			arg_type = Float64
		"--gclip2"
			help = "gradient clip for reinforce"
			default = 100.0
			arg_type = Float64
		"--pg"
			help = "allow policy gradient tuning"
			default = 0
			arg_type = Int
		"--pgbatch"
			help = "batch size for pg"
			default = 10
			arg_type = Int
		"--log"
			help = "name of the log file"
			default = "test.log"
		"--iterative"
			help = "sp + pg"
			default = 1
			arg_type = Int
		"--gamma"
			help = "discount factor"
			default = 0.9
			arg_type = Float64
		"--mom"
			help = "gamma for momentum"
			default = 0.9
			arg_type = Float64
		"--opt"
			help = "adam or momentum"
			default = "adam"
		"--order"
			help = "map orders"
			nargs = '+'
			default = [1, 2, 3]
			arg_type = Int
		"--save"
			help = "model path"
			default = "modelbank/modelacl"
		"--patience1"
			help = "patience param"
			default = 20
			arg_type = Int
		"--patience2"
			help = "patience param"
			default = 50
			arg_type = Int
		"--tunefor"
			help = "tune for (single or paragraph)"
			default = "single"
		"--load"
			help = "model path"
			default = ""
		"--vDev"
			help = "vDev or vTest"
			action = :store_true
		"--pretrain"
			help = "number of additional paragraphs"
			arg_type = Int
			default = 0
		"--charenc"
			help = "charecter embedding"
			action = :store_true
		"--encoding"
			help = "grid or multihot"
			default = "grid"
		"--greedy"
			help = "deterministic or stochastic policy"
			action = :store_true
		"--seed"
			help = "seed number"
			arg_type = Int
			default = 12345

	end
	return parse_args(s)
end		

args = parse_commandline()

include(args["model"])

function execute(train_ins, test_ins, maps, vocab, args; dev_ins=nothing)
	data = map(x -> build_instance(x, maps[x.map], vocab; encoding=args["encoding"]), vcat(train_ins[1], train_ins[2]))
	trn_data = minibatch(data;bs=args["bs"])

	vdims = size(trn_data[1][2][1])
	
	info("\nWorld: $(vdims)")

	w = nothing
	if args["load"] != ""
		w = loadmodel(args["load"])
	else
		if length(vdims) > 2
			w = initweights(KnetArray, args["hidden"], length(vocab)+1, args["embed"], args["window"], vdims[3], args["filters"])
		else
			w = initweights(KnetArray, args["hidden"], length(vocab)+1, args["embed"], vdims[2])
		end
	end

	info("Model Prms:")
	for k in keys(w); info("$k : $(size(w[k])) "); end

	dev_data = dev_ins != nothing ? map(ins -> (ins, ins_arr(vocab, ins.text)), dev_ins) : nothing
	dev_data_grp = dev_ins != nothing ? map(x -> map(ins-> (ins, ins_arr(vocab, ins.text)),x), group_singles(dev_ins)) : nothing

	test_data = map(ins-> (ins, ins_arr(vocab, ins.text)), test_ins)
	#test_data_prg = map(ins-> (ins, ins_arr(d["vocab"], ins.text)), merge_singles(test_ins))
	test_data_grp = map(x->map(ins-> (ins, ins_arr(vocab, ins.text)),x), group_singles(test_ins))

	#prms_sp = initparams(w; args=args)
	#prms_rl = initparams(w; args=args)

	globalbest = 0.0

	for it=1:args["iterative"]
		prms_sp = initparams(w; args=args)
		patience = 0
		sofarbest = 0.0
		for i=1:args["epoch"]
			shuffle!(trn_data)
			@time lss = train(w, prms_sp, trn_data; args=args)
			@time tst_acc = test([w], test_data, maps; args=args)
			@time tst_prg_acc = test_paragraph([w], test_data_grp, maps; args=args)
			#@time trnloss = train_loss(w, trn_data; args=args)
			
			dev_acc = 0
			dev_prg_acc = 0

			if args["vDev"]
				@time dev_acc = test([w], dev_data, maps; args=args)
				@time dev_prg_acc = test_paragraph([w], dev_data_grp, maps; args=args)
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
					savemodel(w, args["save"])
				end
			else
				patience += 1
			end
			
			if args["vDev"]
				info("Epoch: $(i), trn loss: $(lss), single acc: $(dev_acc), paragraph acc: $(dev_prg_acc), $(dev_ins[1].map)")
				info("TestIt: $(i), trn loss: $(lss), single acc: $(tst_acc), paragraph acc: $(tst_prg_acc), $(test_ins[1].map)")
			else
				info("Epoch: $(i), trn loss: $(lss), single acc: $(tst_acc), paragraph acc: $(tst_prg_acc), $(test_ins[1].map)")
			end

			#info("ep: $(i), trn loss: $(trnloss)")
			
			if patience >= args["patience1"]
				break
			end
		end
		
		sofarbest = 0.0
		if args["pg"] != 0
			prms_rl = initparams(w; args=args)
			train_data = map(ins-> (ins, ins_arr(vocab, ins.text)), train_ins[1])
			append!(train_data, map(ins-> (ins, ins_arr(vocab, ins.text)), train_ins[2]))
	
			for i=1:args["pg"]
				@time lss = train_pg(w, prms_rl, train_data, maps; args=args)
				@time tst_acc = test([w], test_data, maps; args=args)
				@time tst_prg_acc = test_paragraph([w], test_data_grp, maps; args=args)
				#@time trnloss = train_loss(w, trn_data; args=args)

				dev_acc = 0
				dev_prg_acc = 0

				if args["vDev"]
					@time dev_acc = test([w], dev_data, maps; args=args)
					@time dev_prg_acc = test_paragraph([w], dev_data_grp, maps; args=args)
				end
				
				if args["vDev"]
					info("PGEpoch: $(i), avg total_rewards: $(lss), single acc: $(dev_acc), paragraph acc: $(dev_prg_acc), $(dev_ins[1].map)")
					info("PGTestIt: $(i), avg total_rewards: $(lss), single acc: $(tst_acc), paragraph acc: $(tst_prg_acc), $(test_ins[1].map)")
				else
					info("PGEpoch: $(i), avg total_rewards: $(lss), single acc: $(tst_acc), paragraph acc: $(tst_prg_acc), $(test_ins[1].map)")
				end

				#info("PGep: $(i), trn loss: $(trnloss)")

				tunefor = args["tunefor"] == "single" ? tst_acc : tst_prg_acc
				tunefordev = args["tunefor"] == "single" ? dev_acc : dev_prg_acc
				tunefor = args["vDev"] ? tunefordev : tunefor

				if tunefor > sofarbest
					sofarbest = tunefor
					patience = 0
					if sofarbest > globalbest
						globalbest = sofarbest
						info("Saving the model...")
						savemodel(w, args["save"])
					end
				else
					patience += 1
				end
				
				if patience >= args["patience2"]
					break
				end

			end
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

function main()
	Logging.configure(filename=args["log"])
	Logging.configure(level=INFO)
	srand(args["seed"])
	info("*** Parameters ***")
	for k in keys(args); info("$k -> $(args[k])"); end
	
	grid, jelly, l = getallinstructions()
	lg = length(grid)
	lj = length(jelly)
	ll = length(l)
	dg = args["vDev"] ? floor(Int, lg*0.1) : 0
	dj = args["vDev"] ? floor(Int, lj*0.1) : 0
	dl = args["vDev"] ? floor(Int, ll*0.1) : 0

	trainins = [(grid[(dg+1):end], jelly[(dj+1):end]), (grid[(dg+1):end], l[(dl+1):end]), (jelly[(dj+1):end], l[(dl+1):end])]
	testins = [l, jelly, grid]
	devins = args["vDev"] ? [vcat(grid[1:dg], jelly[1:dj]), vcat(grid[1:dg], l[1:dg]), vcat(jelly[1:dj], l[1:dl])] : [nothing, nothing, nothing]
	maps = get_maps()

	vocab = !args["charenc"] ? build_dict(vcat(grid, jelly, l)) : build_char_dict(voc_ins)
	info("\nVocab: $(length(vocab))")

	base_s = args["save"]
	base_l = args["load"]
	for i in args["order"]
		args["save"] = string(base_s, "_", args["trainfiles"][i])
		args["load"] = base_l != "" ? string(base_l, "_", args["trainfiles"][i]) : ""
		execute(trainins[i], testins[i], maps, vocab, args; dev_ins=devins[i])
	end
end

main()

