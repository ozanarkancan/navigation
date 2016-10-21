using GPUChecker, CUDArt
CUDArt.device(first_min_used_gpu())

include("util.jl")
include("model01.jl")
using ArgParse, JLD, Hyperopt

function objective(params)
	global args
	global d
	global test_data
	vdims = size(d["data"][1][2][1])
	
	hidden, window, filters = params
	println("\nHidden: $hidden , Window: $window , Filters: $filters")
	
	net = compile(:model; hidden=hidden, embed=hidden, dims=(window, window, vdims[3], filters))
	setp(net, adam=true, lr=args["lr"])
	
	best = 10.0

	for i=1:args["epoch"]
		@time lss = train(net, d["data"])
		@time tst_acc = test(net, test_data, d["maps"])
		if i % 5 == 0
			gc()
		end
		#tst_acc = 0.0

		println("Epoch: $(i), trn loss: $(lss), tst acc: $(tst_acc)")
		flush(STDOUT)

		if (1-tst_acc) < best
			best = 1-tst_acc
		end
	end
	return Dict("loss" => best, "status" => STATUS_OK)
end

function main()
	global args = parse_commandline()
	println("*** Parameters ***")
	for k in keys(args); println("$k -> $(args[k])"); end
	flush(STDOUT)

	global d = load(args["trainfile"])

	test_ins = getinstructions(args["testfile"])
	global test_data = map(ins-> (ins, ins_arr(d["vocab"], ins.text)), test_ins)

	hidden_list = [32,64,128]
	window_list = collect(3:2:39)
	filter_list = [10, 20, 50, 100, 200, 300, 400]

	best = fmin(objective,
		space=[choice("hidden", hidden_list), choice("window", window_list), choice("filters", filter_list)],
		algo=TPESUGGEST,
		max_evals=args["maxevals"])
	
	println("Best:")
	println("Hidden: $(hidden_list[best["hidden"] + 1])")
	println("Window: $(window_list[best["window"] + 1])")
	println("Filters: $(filter_list[best["filters"] + 1])")
end

#=
#history
function train(net, data; gclip=10.0)
	lss = 0.0
	cnt = 0.0
	dummy = CudaArray(zeros(Float64, 4, 1))
	for (ins, states, Y) in data
		for w=1:length(ins); sforw(net, ins[w], dummy; dropout=true); end
		for i=1:length(states); ypred = sforw(net, states[i], (i==1 ? dummy : Y[i-1]); decode=true, dropout=true); lss += softloss(ypred, Y[i]); cnt += 1; end
		for i=length(Y):-1:1; sback(net, Y[i], softloss); end
		for w in ins; sback(net); end

		update!(net; gclip = gclip)
		reset!(net)
	end

	return lss / cnt
end

function test(net, data, maps; limactions=35)
	scss = 0.0

	dummy = CudaArray(zeros(Float64, 4, 1))
	for (instruction, wordmatrices) in data
		for w=1:length(wordmatrices); forw(net, wordmatrices[w], dummy); end

		current = instruction.path[1]
		nactions = 0
		stop = false
		
		println("\n$(instruction.text)")
		println("Path: $(instruction.path)")
		actions = Any[]
		prevAction = zeros(Float64, 4, 1)
		while !stop
			state = state_agent_centric(maps[instruction.map], current)
			ypred = forw(net, state, prevAction; decode=true)
			action = indmax(to_host(ypred))
			prevAction[:] = 0.0
			prevAction[action] = 1.0
			push!(actions, action)
			current = getlocation(maps[instruction.map], current, action)
			nactions += 1

			stop = nactions > limactions || action == 4 || !haskey(maps[instruction.map].nodes, (current[1], current[2]))
		end
		println("Actions: $(reshape(collect(actions), 1, length(actions)))")
		println("Current: $(current)")

		scss =  current == instruction.path[end] ? scss + 1 : scss

		reset!(net)
	end

	return scss / length(data)
end
=#

#=attention
function train(net, data; gclip=10.0)
	lss = 0.0
	cnt = 0.0
	for (ins, states, Y) in data
		for i=1:length(states)
			for w=1:length(ins); sforw(net, ins[w]; dropout=true); end
			ypred = sforw(net, states[i]; decode=true, dropout=true)
			lss += softloss(ypred, Y[i])
			cnt += 1
		end
		for i=length(Y):-1:1
			sback(net, Y[i], softloss)
			for w in ins; sback(net); end;
		end

		update!(net; gclip = gclip)
		reset!(net)
	end

	return lss / cnt
end

function test(net, data, maps; limactions=35)
	scss = 0.0

	for (instruction, wordmatrices) in data
		current = instruction.path[1]
		nactions = 0
		stop = false
		
		println("\n$(instruction.text)")
		println("Path: $(instruction.path)")
		actions = Any[]

		while !stop
			for w=1:length(wordmatrices); forw(net, wordmatrices[w]); end
			state = state_agent_centric(maps[instruction.map], current)
			ypred = forw(net, state; decode=true)
			action = indmax(to_host(ypred))
			push!(actions, action)
			current = getlocation(maps[instruction.map], current, action)
			nactions += 1

			stop = nactions > limactions || action == 4 || !haskey(maps[instruction.map].nodes, (current[1], current[2]))
		end
		println("Actions: $(reshape(collect(actions), 1, length(actions)))")
		println("Current: $(current)")

		scss =  current == instruction.path[end] ? scss + 1 : scss

		reset!(net)
	end

	return scss / length(data)
end
=#

#dropout
function train(net, data; gclip=10.0)
	lss = 0.0
	cnt = 0.0
	for (ins, states, Y) in data
		for w=1:length(ins); sforw(net, ins[w]; dropout=true); end
		for i=1:length(states); ypred = sforw(net, states[i]; decode=true, dropout=true); lss += softloss(ypred, Y[i]); cnt += 1; end
		for i=length(Y):-1:1; sback(net, Y[i], softloss); end
		for w in ins; sback(net); end

		update!(net; gclip = gclip)
		reset!(net)
	end

	return lss / cnt
end

function test(net, data, maps; limactions=35)
	scss = 0.0

	for (instruction, wordmatrices) in data
		for w=1:length(wordmatrices); forw(net, wordmatrices[w]); end

		current = instruction.path[1]
		nactions = 0
		stop = false
		
		println("\n$(instruction.text)")
		println("Path: $(instruction.path)")
		actions = Any[]

		while !stop
			state = state_agent_centric(maps[instruction.map], current)
			ypred = forw(net, state; decode=true)
			action = indmax(to_host(ypred))
			push!(actions, action)
			current = getlocation(maps[instruction.map], current, action)
			nactions += 1

			stop = nactions > limactions || action == 4 || !haskey(maps[instruction.map].nodes, (current[1], current[2]))
		end
		println("Actions: $(reshape(collect(actions), 1, length(actions)))")
		println("Current: $(current)")

		if current == instruction.path[end]
			scss += 1
			println("SUCCESS")
		else
			println("FAILURE")
		end

		flush(STDOUT)
		
		reset!(net)
	end

	return scss / length(data)
end

#=
function train(net, data; gclip=10.0)
	lss = 0.0
	cnt = 0.0
	for (ins, states, Y) in data
		for w=1:length(ins); sforw(net, ins[w]); end
		for i=1:length(states); ypred = sforw(net, states[i]; decode=true); lss += softloss(ypred, Y[i]); cnt += 1; end
		for i=length(Y):-1:1; sback(net, Y[i], softloss); end
		for w in ins; sback(net); end

		update!(net; gclip = gclip)
		reset!(net)
	end

	return lss / cnt
end

function test(net, data, maps; limactions=35)
	scss = 0.0

	for (instruction, wordmatrices) in data
		for w=1:length(wordmatrices); forw(net, wordmatrices[w]); end

		current = instruction.path[1]
		nactions = 0
		stop = false
		
		println("\n$(instruction.text)")
		println("Path: $(instruction.path)")
		actions = Any[]

		while !stop
			state = state_agent_centric(maps[instruction.map], current)
			ypred = forw(net, state; decode=true)
			action = indmax(to_host(ypred))
			push!(actions, action)
			current = getlocation(maps[instruction.map], current, action)
			nactions += 1

			stop = nactions > limactions || action == 4 || !haskey(maps[instruction.map].nodes, (current[1], current[2]))
		end
		println("Actions: $(reshape(collect(actions), 1, length(actions)))")
		println("Current: $(current)")

		scss =  current == instruction.path[end] ? scss + 1 : scss

		reset!(net)
	end

	return scss / length(data)
end
=#

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin
		"--lr"
			help = "learning rate"
			default = 0.001
			arg_type = Float64
		"--epoch"
			help = "number of epochs"
			default = 100
			arg_type = Int
		"--maxevals"
			help = "number of evals"
			default = 10
			arg_type = Int
		"--trainfile"
			help = "built training jld file"
			default = "grid_l.jld"
		"--testfile"
			help = "test file as regular instruction file(json)"
			default = "data/instructions/SingleSentenceZeroInitial.jelly.json"
	end
	return parse_args(s)
end		

main()
