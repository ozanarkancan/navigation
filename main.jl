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
		"--trainfile"
			help = "built training jld file"
			default = "grid_l.jld"
		"--testfile"
			help = "test file as regular instruction file(json)"
			default = "data/instructions/SingleSentenceZeroInitial.jelly.json"
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
		"--bs"
			help = "batch size"
			default = 100
			arg_type = Int
	end
	return parse_args(s)
end		

args = parse_commandline()

include(args["model"])

function main()
	println("*** Parameters ***")
	for k in keys(args); println("$k -> $(args[k])"); end
	flush(STDOUT)

	d = load(args["trainfile"])
	trn_data = minibatch(d["data"];bs=args["bs"])
	vdims = size(trn_data[1][2][1])

	w = initweights(KnetArray, args["hidden"], length(d["vocab"])+1, args["embed"], 0.1, args["window"], vdims[3], args["filters"])
	prms = initparams(w; lr=args["lr"])
	
	test_ins = getinstructions(args["testfile"])
	test_data = map(ins-> (ins, ins_arr(d["vocab"], ins.text)), test_ins)
	#test_data = map(ins-> (ins, ins_char_arr(d["vocab"], ins.text)), test_ins)

	for i=1:args["epoch"]
		@time lss = train(w, prms, trn_data; args=args)
		@time tst_acc = test(w, test_data, d["maps"]; args=args)
		
		println("Epoch: $(i), trn loss: $(lss), tst acc: $(tst_acc)")
		flush(STDOUT)
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
