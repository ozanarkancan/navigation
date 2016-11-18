include("util.jl")
using ArgParse

function results(fname)
	d = Dict()
	s = readstring(pipeline(`less $fname`, `grep Epoch`))
	lns = split(s, "\n")
	for line in lns
		if line == ""
			continue;
		end
		l = split(line)

		v = parse(Float64, split(l[8], ",")[1])
		if haskey(d, l[9])
			if v > d[l[9]]
				d[l[9]] = v
			end
		else
			d[l[9]] = v
		end
	end
	return mean(values(d)), d
end

function main()
	args = parse_commandline()

	bestr = 0.0
	bestf = ""
	bestd = nothing

	files = readdir(args["folder"])

	for f in files
		try
			r, d = results(string(args["folder"], "/", f))
			if r > bestr
				bestr = r
				bestf = f
				bestd = d
			end
		catch e
			println(e)
			println("Error on $f")
		end
	end
	
	println("Best file: $bestf , Results: $bestr")
	for k in keys(bestd); println("$k : $(bestd[k])"); end
end

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin
		"--folder"
			help = "folder contains the log files"
			default = "logs"
	end
	return parse_args(s)
end		

main()
