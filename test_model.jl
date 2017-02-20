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
		"--log"
			help = "name of the log file"
			default = "test.log"
		"--test"
			help = "1,2 or 3 (l, jelly, grid)"
			arg_type = Int
		"--load"
			help = "model path"
			default = []
			nargs = '+'
		"--charenc"
			help = "charecter embedding"
			action = :store_true
		"--encoding"
			help = "grid or multihot"
			default = "grid"
		"--greedy"
			help = "deterministic or stochastic policy"
			action = :store_true
		"--embedding"
			help = "embedding"
			action = :store_true
		"--seed"
			help = "seed"
			default = 12345
			arg_type = Int
		"--vTest"
			help = "vtest"
			action = :store_true
        "--categorical"
            help = "report action categories"
            action = :store_true
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
	Logging.configure(filename=args["log"])
	Logging.configure(level=INFO)
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

	testins = args["vTest"] ? [l, jelly, grid] : [l[(dl+1):end], l[1:dl], jelly[(dj+1):end], jelly[1:dj], grid[(dg+1):end], grid[1:dg]]
	maps = get_maps()

	vocab = !args["charenc"] ? build_dict(vcat(grid, jelly, l)) : build_char_dict(voc_ins)
	info("\nVocab: $(length(vocab))")
	emb = load("data/embeddings.jld", "vectors")

	models = Any[]
	for mfile in args["load"]; push!(models, loadmodel(mfile)); end

	test_ins = testins[args["test"]]
	test_data = args["embedding"] ? map(ins-> (ins, ins_arr_embed(emb, vocab, ins.text)), test_ins) : map(ins-> (ins, ins_arr(vocab, ins.text)), test_ins)
	test_data_grp = args["embedding"] ? map(x->map(ins-> (ins, ins_arr_embed(emb, vocab, ins.text)),x), group_singles(test_ins)) : map(x->map(ins-> (ins, ins_arr(vocab, ins.text)),x), group_singles(test_ins))

	@time tst_acc = test(models, test_data, maps; args=args)
	@time tst_prg_acc = test_paragraph(models, test_data_grp, maps; args=args)

	info("Single: $tst_acc , Paragraph: $tst_prg_acc")

    if args["categorical"]
        df = DataFrame(Category=Any[], Acc=AbstractFloat[], Text=Any[], Path=Any[], Map=Any[], Fname=Any[], Id=Any[])
        cat = Dict()
        num = Dict()
        for i=1:length(test_data)
            as = getactions(test_data[i][1].path)
            acc = test(models, test_data[i:i], maps;args=args)
            push!(df, (as, acc, join(test_data[i][1].text, " "), test_data[i][1].path, test_data[i][1].map, test_data[i][1].fname, test_data[i][1].id))
            if haskey(cat, as)
                cat[as] = cat[as] + acc
                num[as] = num[as] + 1
            else
                cat[as] = acc
                num[as] = 1
            end
        end

        writetable(string(args["log"], ".csv"), df)

        df2 = DataFrame(Category=Any[], Acc=AbstractFloat[], Num=Int[])
        for k in sort(collect(keys(cat)); by=length)
            info("\nCategory: $k , Dist: $(num[k]) / $(length(test_data))")
            info("Accuracy: $(cat[k] / num[k])")
            push!(df2, (k, cat[k]/num[k], num[k]))
        end
        writetable(string(args["log"], "_2.csv"), df2)
    end
end

main()
