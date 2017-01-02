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

		i = 3
		stop = false
		while !stop
			if startswith(l[i], "acc")
				stop = true
			end
			i += 1
		end

		v = parse(Float64, split(l[i], ",")[1])
		if haskey(d, l[end])
			if v > d[l[end]]
				d[l[end]] = v
			end
		else
			d[l[end]] = v
		end
	end
	return mean(values(d)), d
end

function printlogs(logs)
	for (r,d,f) in logs
		println("File: $f , Results: $r")
		for k in keys(d); println("$k : $(d[k])"); end
		println("")
	end
end

function main()
	args = parse_commandline()

	files = readdir(args["folder"])
	logs = Any[]

	for f in files
		try
			r, d = results(string(args["folder"], "/", f))
			push!(logs, (r, d, f))

		catch e
			println(e)
			println("Error on $f")
		end
	end

	sort!(logs, rev=true)
	printlogs(logs)
	
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
