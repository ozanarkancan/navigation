using ArgParse

include("util.jl")

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
		"--bs"
			help = "batch size"
			default = 100
			arg_type = Int
		"--gclip"
			help = "gradient clip"
			default = 10.0
			arg_type = Float64
	end
	return parse_args(s)
end		

args = parse_commandline()

include(args["model"])

function execute(trainfile, test_ins, args)
	d = load(trainfile)
	trn_data = minibatch(d["data"];bs=args["bs"])
	vdims = size(trn_data[1][2][1])

	println("Vocab: $(length(d["vocab"])), World: $(vdims)")
	
	w = nothing
	if length(vdims) > 2
		w = initweights(KnetArray, args["hidden"], length(d["vocab"])+1, args["embed"], args["window"], vdims[3], args["filters"])
	else
		w = initweights(KnetArray, args["hidden"], length(d["vocab"])+1, args["embed"], vdims[2])
	end

	println("Model Prms:")
	for k in keys(w); println("$k : $(size(w[k])) "); end

	prms = initparams(w; lr=args["lr"])
	
	test_data = map(ins-> (ins, ins_arr(d["vocab"], ins.text)), test_ins)
	#test_data = map(ins-> (ins, ins_char_arr(d["vocab"], ins.text)), test_ins)

	for i=1:args["epoch"]
		shuffle!(trn_data)
		@time lss = train(w, prms, trn_data; args=args)
		@time tst_acc = test(w, test_data, d["maps"]; args=args)
		
		println("Epoch: $(i), trn loss: $(lss), tst acc: $(tst_acc), $(test_ins[1].map)")
		flush(STDOUT)
	end
end

function main()
	srand(12345)
	println("*** Parameters ***")
	for k in keys(args); println("$k -> $(args[k])"); end
	flush(STDOUT)

	grid, jelly, l = getallinstructions()
	testins = [l, jelly, grid]

	for i=1:length(args["trainfiles"])
		execute(args["trainfiles"][i], testins[i], args)
	end
end

main()

#=
MODELS

** model 01 **
- Encoder-Decoder
- 2 layers (both encoder & decoder)
- 0.2 dropout after the word embeddings (there is no dropout after the embedding of the perception input)
- 0.5 dropout between lstm layers
- embedding of the perception input: conv

** model 02 **
- Encoder-Decoder
- 2 layers (both encoder & decoder)
- 0.2 dropout after the word embeddings & the embedding of the perception input
- 0.5 dropout between lstm layers

** model 03 **
- Same as model01 except
- embedding of the perception input: relu . conv

** model 04 **
- Same as model01 except
- Final output is produced by using embedding of the perception input and the hidden of the final layer

** model 05 **
- Use embeddings in the internal layers

** model 06 **
- Same as model01 except
- Lstm uses both t-1 and t-2 hiddens

** model 07 **
- Bidirectional encoder
- Decoder takes the transformation of the concatenated f-b last hiddens as input

** model 08 **
- sigmoid is used after the conv
=#
