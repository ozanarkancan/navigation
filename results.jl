include("util.jl")
using ArgParse, DataFrames, Query

MapSize = Dict("Grid"=>874, "Jelly"=>1293, "L"=>1070, "Total"=>3237)
MapSizeP = Dict("Grid"=>224, "Jelly"=>242, "L"=>236, "Total"=>702)

function log2csv(fname)
	df = DataFrame(epochtype = [], epoch = [], loss_reward = [], single = [], paragraph = [], map = [])
	s = readstring(pipeline(`less $fname`, `grep Epoch`))
	lns = split(s, "\n")

	for line in lns
		if line == ""
			continue
		end
		l = split(line)

		et = split(l[2], ":")[end-1]
		ep = parse(Int, split(l[3], ",")[1])
		ls = parse(Float32, split(l[6], ",")[1])
		sacc = parse(Float32, split(l[9], ",")[1])
		pacc = parse(Float32, split(l[12], ",")[1])

		push!(df, (et, ep, ls, sacc, pacc, l[end]))
	end
	return df
end


function results(fname, sortby)
	d = Dict()
	s = readstring(pipeline(`less $fname`, `grep Epoch`))
	lns = split(s, "\n")
	for line in lns
		if line == ""
			continue;
		end
		l = split(line)

		i = 5
		stop = false
		while !stop
			if startswith(l[i], sortby)
				stop = true
				i += 2
			else
				i += 1
			end
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

	avg = 0.0
	for k in keys(d)
		avg += (d[k]*MapSize[k])/MapSize["Total"]
	end
	return avg, d
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

	overview = DataFrame(fname=[], singlebest=[], paragraph=[], single=[], paragraphbest=[], scores=[])

	for f in files
		try
			df = log2csv(string(args["folder"], "/", f))
			writetable(string("experiments/", f, ".csv"), df)
			
			scores = Any[]

			for m in ["Grid", "Jelly", "L"]
				q1 = @from i in df begin
					@where i.map == m
					@orderby descending(i.single)
					@select i
					@collect DataFrame
				end

				q2 = @from i in df begin
					@where i.map == m
					@orderby descending(i.paragraph)
					@select i
					@collect DataFrame
				end

				push!(scores, (q1[1, 4], q1[1, 5], q2[1, 4], q2[1, 5]))
			end

			weighteds = map(tup->map(x->x*MapSize[tup[2]]/MapSize["Total"], tup[1][1:2:4]), collect(zip(scores, ["Grid", "Jelly", "L"])))
			weightedp = map(tup->map(x->x*MapSizeP[tup[2]]/MapSizeP["Total"], tup[1][2:2:4]), collect(zip(scores, ["Grid", "Jelly", "L"])))
			push!(overview, (f,
			sum(map(x->x[1], weighteds)),
			sum(map(x->x[1], weightedp)),
			sum(map(x->x[2], weighteds)),
			sum(map(x->x[2], weightedp)),
			scores
			))
		catch e
			println(e)
			println("Error on $f")
		end
	end
	sort!(overview, rev=true, cols = [:singlebest])
	println(overview)
	writetable("overview.csv", overview)
end

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin
		"--folder"
			help = "folder contains the log files"
			default = "logs"
		"--sortby"
			help = "single or paragraph"
			default = "single"
	end
	return parse_args(s)
end		

main()
