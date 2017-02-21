using ArgParse, DataFrames

include("util.jl")
include("io.jl")

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin
		"--hidden"
			help = "hidden size"
			default = 100
			arg_type = Int
		"--embed"
			help = "embedding size"
			default = 100
			arg_type = Int
		"--limactions"
			arg_type = Int
			default = 35
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
			default = [29, 7, 5]
			arg_type = Int
			nargs = '+'
		"--filters"
			help = "number of filters"
			default = [300, 150, 50]
			arg_type = Int
			nargs = '+'
		"--model"
			help = "model file"
			default = "baseline_cnn_wvecs.jl"
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
			default = 1
			arg_type = Int
		"--log"
			help = "name of the log file"
			default = "test.log"
		"--test"
			help = "1,2 or 3 (l, jelly, grid)"
			arg_type = Int
		"--load"
			help = "model path"
			default = ["mbank/cnn_wvecs_3_1_l_jelly.jld"]
			nargs = '+'
		"--charenc"
			help = "charecter embedding"
			action = :store_true
		"--encoding"
			help = "grid or multihot"
			default = "grid"
		"--greedy"
			help = "deterministic or stochastic policy"
			action = :store_false
		"--embedding"
			help = "embedding"
			action = :store_false
		"--seed"
			help = "seed"
			default = 12345
			arg_type = Int
        "--mapname"
            help = "name of the map"
            default = "Grid"
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

function main()
	Logging.configure(output=STDOUT)
	Logging.configure(level=INFO)
	srand(args["seed"])
	info("*** Parameters ***")
	for k in keys(args); info("$k -> $(args[k])"); end

    info("Loading...")
	grid, jelly, l = getallinstructions()

	maps = get_maps()

	vocab = !args["charenc"] ? build_dict(vcat(grid, jelly, l)) : build_char_dict(voc_ins)
	info("\nVocab: $(length(vocab))")
	emb = load("data/embeddings.jld", "vectors")

	models = Any[]
	for mfile in args["load"]; push!(models, loadmodel(mfile)); end

    info("Working on $(args["mapname"]) map...")

    while true
        try
            print("Coordinates and the orientation(x,y,o): ")
            str = readline(STDIN)
            x,y,o = map(x->parse(Int, x), split(strip(str), ","))

            print("Enter an instruction: ")
            str = readline(STDIN)
            text = split(strip(str))

            ins = Instruction("demo", text, Any[(x,y,o)], args["mapname"], 0)
            dat = args["embedding"] ? [(ins, ins_arr_embed(emb, vocab, ins.text))] : [(ins, ins_arr(vocab, ins.text))]

            test(models, dat, maps; args=args)

            print("Continue y/n: ")
            c = readline(STDIN)
            c = strip(c)
            if c=="N" || c=="n" || c=="no" || c=="No"
                break
            end
        catch e
            info(e)
            info("Bad things happened...\n")
        end
    end
end

main()
