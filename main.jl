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
			default = 3
			arg_type = Int
		"--filters"
			help = "number of filters"
			default = 30
			arg_type = Int
		"--model"
			help = "model file"
			default = "model01.jl"
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
	vdims = size(d["data"][1][2][1])
	
	if args["model"] == "model07.jl"
		encoder = compile(:encoder; hidden=args["hidden"], embed=args["embed"])
		decoder = compile(:decoder; hidden=args["hidden"], dims=(args["window"], args["window"], vdims[3], args["filters"]))

		setp(encoder, adam=true, lr=args["lr"])
		setp(decoder, adam=true, lr=args["lr"])
		
		net = (encoder, decoder)
		trnf = (net,data) -> train(net, data; hidden=args["hidden"])
	else
		net = compile(:model; hidden=args["hidden"], embed=args["embed"], dims=(args["window"], args["window"], vdims[3], args["filters"]))
		setp(net, adam=true, lr=args["lr"])
		trnf = train
	end
	
	test_ins = getinstructions(args["testfile"])
	test_data = map(ins-> (ins, ins_arr(d["vocab"], ins.text)), test_ins)
	#test_data = map(ins-> (ins, ins_char_arr(d["vocab"], ins.text)), test_ins)

	for i=1:args["epoch"]
		@time lss = trnf(net, d["data"])
		@time tst_acc = test(net, test_data, d["maps"])
		if i % 5 == 0
			gc()
		end

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
