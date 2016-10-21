include("util.jl")
using ArgParse, JLD

function indices(fname)
	f = open(fname)
	lines = readlines(f)
	close(f)
	
	experiment = Any[]
	epoch = 1
	indx = 0
	corrects = Int[]
	fails = Int[]
	actions = Any[]

	for i=1:length(lines)
		l = lines[i]
		if startswith(l, "SubString")
			indx += 1
			pathE = eval(parse(split(lines[i+1])[2]))[end]
			curr = eval(parse(split(lines[i+3])[end]))
			lst = (pathE[1] == curr[1] && pathE[2] == curr[2] && pathE[3] == curr[3]) ? corrects : fails
			
			if epoch == 91
				println("Curr: $curr , PathE: $pathE, res: $(pathE[1] == curr[1] && pathE[2] == curr[2] && pathE[3] == curr[3])")
			end
			
			push!(lst, indx)
			push!(actions, eval(parse(strip(split(lines[i+2], ":")[end]))))
		elseif startswith(l, "Epoch")
			indx = 0
			push!(experiment, (epoch, copy(corrects), copy(fails), copy(actions), parse(Float64, split(l)[end])))
			epoch += 1
			empty!(corrects)
			empty!(fails)
			empty!(actions)
		end
	end

	return experiment
end

function main()
	args = parse_commandline()

	expr = indices(args["logname"])
	sorted = sort!(expr, by=x->x[5])
	println(sorted[end])
end

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin
		"--logname"
			help = "log file"

		"--testfile"
			help = "test file"

		"--outname"
			default = ""
	end
	return parse_args(s)
end		

#main()
