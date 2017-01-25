using Knet, AutoGrad, Logging

include("inits.jl")

function spatial(emb, x)
	return x * emb
end

function lstm(weight,bias,hidden,cell,input)
	gates   = hcat(input,hidden) * weight .+ bias
	hsize   = size(hidden,2)
	forget  = sigm(gates[:,1:hsize])
	ingate  = sigm(gates[:,1+hsize:2hsize])
	outgate = sigm(gates[:,1+2hsize:3hsize])
	change  = tanh(gates[:,1+3hsize:end])
	cell    = cell .* forget + ingate .* change
	hidden  = outgate .* tanh(cell)
	return (hidden,cell)
end

function lstm2(weight,bias,hidden,cell,input,att)
	h = hcat(input, hcat(hidden, att))
	gates   = h * weight .+ bias
	hsize   = size(hidden,2)
	forget  = sigm(gates[:,1:hsize])
	ingate  = sigm(gates[:,1+hsize:2hsize])
	outgate = sigm(gates[:,1+2hsize:3hsize])
	change  = tanh(gates[:,1+3hsize:end])
	cell    = cell .* forget + ingate .* change
	hidden  = outgate .* tanh(cell)
	return (hidden,cell)
end


function encode(weight1_f, bias1_f, weight1_b, bias1_b, emb, state, words; dropout=false, pdrops=[0.5, 0.5])
	for i=1:length(words)
		x = words[i] * emb

		if dropout && pdrops[1] > 0.0
			x = x .* (rand!(similar(AutoGrad.getval(x))) .> pdrops[1]) * (1/(1-pdrops[1]))
		end

		state[1][i+1], state[2][i+1] = lstm(weight1_f, bias1_f, state[1][i], state[2][i], x)


		x = words[end-i+1] * emb

		if dropout && pdrops[1] > 0.0
			x = x .* (rand!(similar(AutoGrad.getval(x))) .> pdrops[1]) * (1/(1-pdrops[1]))
		end

		state[3][i+1], state[4][i+1] = lstm(weight1_b, bias1_b, state[3][i], state[4][i], x)
	end
end

function decode(weight1, bias1, soft_w1, soft_w2, soft_w3, soft_b, state, x,
	mask; dropout=false, pdrops=[0.5, 0.5, 0.5])
	if dropout && pdrops[1] > 0.0
		x = x .* (rand!(similar(AutoGrad.getval(x))) .> pdrops[1]) * (1/(1-pdrops[1]))
	end

	state[5], state[6] = lstm(weight1, bias1, state[5], state[6], x)

	inp = state[5]
	if dropout && pdrops[2] > 0.0
		inp = inp .* (rand!(similar(AutoGrad.getval(inp))) .> pdrops[2]) * (1/(1-pdrops[2]))
		#state[6] = state[6] .* (rand!(similar(AutoGrad.getval(state[6]))) .> pdrops[2]) * (1/(1-pdrops[2]))
	end

	#q = (state[6] * soft_w1) + x + (att * soft_w2)
	q = (inp * soft_w1) + x * soft_w2 .+ soft_b

	return q
	#return q * soft_w3 .+ soft_b
	#return q * soft_w3
end


function sample(linear)
	linear = linear .- maximum(linear, 2)
	probs = exp(linear) ./ sum(exp(linear), 2)
	info("Probs: $probs")
	c_probs = cumsum(probs, 2)
	return indmax(c_probs .> rand())
end

function discount(rewards; γ=0.9)
	discounted = zeros(Float32, length(rewards), 1)
	discounted[end] = rewards[end]

	for i=(length(rewards)-1):-1:1
		discounted[i] = rewards[i] + γ * discounted[i+1]
	end
	return discounted
end

function loss(weights, state, words, views, ys, maskouts;lss=nothing, dropout=false, pdrops=[0.5, 0.5, 0.5])
	total = 0.0; count = 0

	#encode
	encode(weights["enc_w1_f"], weights["enc_b1_f"], weights["enc_w1_b"], weights["enc_b1_b"],
		weights["emb_word"], state, words; dropout=dropout, pdrops=pdrops)

	state[5] = hcat(state[1][end], state[3][end])
	state[6] = hcat(state[2][end], state[4][end])
	
	#decode
	for i=1:length(views)
		x = spatial(weights["emb_world"], views[i])
		
		ypred = decode(weights["dec_w1"], weights["dec_b1"], weights["soft_w1"], weights["soft_w2"], 
			weights["soft_w3"], weights["soft_b"], state, x, maskouts[i];
			dropout=dropout, pdrops=pdrops)

		ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
		total += sum((ys[i] .* ynorm) .* maskouts[i])
		count += sum(maskouts[i])
	end

	nll = -total/count
	lss[1] = AutoGrad.getval(nll)
	lss[2] = AutoGrad.getval(count)
	return nll
end

lossgradient = grad(loss)

function train(w, prms, data; args=nothing)
	lss = 0.0
	cnt = 0.0
	nll = Float32[0, 0]
	for (words, views, ys, maskouts) in data
		bs = size(words[1], 1)
		state = initstate(KnetArray{Float32}, convert(Int, size(w["enc_b1_f"],2)/4), bs, length(words))

		#load data to gpu
		words = map(t->convert(KnetArray{Float32}, t), words)
		views = map(v->convert(KnetArray{Float32}, v), views)
		ys = map(t->convert(KnetArray{Float32}, t), ys)
		maskouts = map(t->convert(KnetArray{Float32}, t), maskouts)

		g = lossgradient(w, state, words, views, ys, maskouts; lss=nll, dropout=true, pdrops=args["pdrops"])

		gclip = args["gclip"]
		if gclip > 0
			gnorm = 0
			for k in keys(g); gnorm += sumabs2(g[k]); end
			gnorm = sqrt(gnorm)

			debug("Gnorm: $gnorm")

			if gnorm > gclip
				for k in keys(g)
					g[k] = g[k] * gclip / gnorm
				end
			end
		end

		#update weights
		for k in keys(g)
			Knet.update!(w[k], g[k], prms[k])
		end

		lss += nll[1] * nll[2]
		cnt += nll[2]
	end
	return lss / cnt
end

function train_loss(w, data; args=nothing)
	lss = 0.0
	cnt = 0.0
	nll = Float32[0, 0]
	for (words, views, ys, maskouts) in data
		bs = size(words[1], 1)
		state = initstate(KnetArray{Float32}, convert(Int, size(w["enc_b1_f"],2)/4), bs, length(words))

		#load data to gpu
		words = map(t->convert(KnetArray{Float32}, t), words)
		views = map(v->convert(KnetArray{Float32}, v), views)
		ys = map(t->convert(KnetArray{Float32}, t), ys)
		maskouts = map(t->convert(KnetArray{Float32}, t), maskouts)

		loss(w, state, words, views, ys, maskouts; lss=nll, dropout=true, pdrops=args["pdrops"])

		lss += nll[1] * nll[2]
		cnt += nll[2]
	end
	return lss / cnt
end


function test(weights, data, maps; args=nothing)
	scss = 0.0
	mask = convert(KnetArray, ones(Float32, 1,1))

	for (instruction, words) in data
		words = map(v->convert(KnetArray{Float32},v), words)
		state = initstate(KnetArray{Float32}, convert(Int, size(weights["enc_b1_f"],2)/4), 1, length(words))
		
		encode(weights["enc_w1_f"], weights["enc_b1_f"], weights["enc_w1_b"], weights["enc_b1_b"], weights["emb_word"], state, words)
		
		state[5] = hcat(state[1][end], state[3][end])
		state[6] = hcat(state[2][end], state[4][end])
	
		current = instruction.path[1]
		nactions = 0
		stop = false
		
		actions = Any[]

		while !stop
			view = state_agent_centric_multihot(maps[instruction.map], current)
			view = convert(KnetArray{Float32}, view)
			x = spatial(weights["emb_world"], view)

			ypred = decode(weights["dec_w1"], weights["dec_b1"], weights["soft_w1"], weights["soft_w2"], 
			weights["soft_w3"], weights["soft_b"], state, x, mask)
			action = indmax(Array(ypred))
			push!(actions, action)
			current = getlocation(maps[instruction.map], current, action)
			nactions += 1

			stop = nactions > args["limactions"] || action == 4 || !haskey(maps[instruction.map].nodes, (current[1], current[2]))
		end
		
		info("$(instruction.text)")
		info("Path: $(instruction.path)")
		info("Filename: $(instruction.fname)")

		info("Actions: $(reshape(collect(actions), 1, length(actions)))")
		info("Current: $(current)")

		if current == instruction.path[end]
			scss += 1
			info("SUCCESS\n")
		else
			info("FAILURE\n")
		end
	end

	return scss / length(data)
end

function test_paragraph(weights, groups, maps; args=nothing)
	scss = 0.0
	mask = convert(KnetArray, ones(Float32, 1,1))

	for data in groups
		info("\nNew paragraph")
		current = data[1][1].path[1]
		
		for i=1:length(data)
			instruction, words = data[i]
			words = map(v->convert(KnetArray{Float32},v), words)
			state = initstate(KnetArray{Float32}, convert(Int, size(weights["enc_b1_f"],2)/4), 1, length(words))
		
			encode(weights["enc_w1_f"], weights["enc_b1_f"], weights["enc_w1_b"], weights["enc_b1_b"], weights["emb_word"], state, words)
			state[5] = hcat(state[1][end], state[3][end])
			state[6] = hcat(state[2][end], state[4][end])

			nactions = 0
			stop = false
		
			actions = Any[]
			action = 0

			while !stop
				view = state_agent_centric_multihot(maps[instruction.map], current)
				view = convert(KnetArray{Float32}, view)
				x = spatial(weights["emb_world"], view)

				ypred = decode(weights["dec_w1"], weights["dec_b1"], weights["soft_w1"], weights["soft_w2"], 
				weights["soft_w3"], weights["soft_b"], state, x, mask)
				action = indmax(Array(ypred))
				push!(actions, action)
				current = getlocation(maps[instruction.map], current, action)
				nactions += 1

				stop = nactions > args["limactions"] || action == 4 || !haskey(maps[instruction.map].nodes, (current[1], current[2]))
			end
		
			info("$(instruction.text)")
			info("Path: $(instruction.path)")
			info("Filename: $(instruction.fname)")

			info("Actions: $(reshape(collect(actions), 1, length(actions)))")
			info("Current: $(current)")

			if action != 4
				info("FAILURE")
				break
			end

			if i == length(data)
				if current[1] == instruction.path[end][1] && current[2] == instruction.path[end][2]
					scss += 1
					info("SUCCESS\n")
				else
					info("FAILURE\n")
				end
			end
		end
	end

	return scss / length(groups)
end


function initweights(atype, hidden, vocab, embed, onehotworld)
	weights = Dict()
	input = embed
	
	#first layer
	weights["enc_w1_f"] = xavier(Float32, input+hidden, 4*hidden)
	weights["enc_b1_f"] = zeros(Float32, 1, 4*hidden)
	weights["enc_b1_f"][1:hidden] = 1 # forget gate bias
	
	weights["enc_w1_b"] = xavier(Float32, input+hidden, 4*hidden)
	weights["enc_b1_b"] = zeros(Float32, 1, 4*hidden)
	weights["enc_b1_b"][1:hidden] = 1 # forget gate bias

	weights["dec_w1"] = xavier(Float32, input + hidden*2, 4*hidden*2)
	weights["dec_b1"] = zeros(Float32, 1, 4*hidden*2)
	weights["dec_b1"][1:hidden*2] = 1 # forget gate bias

	weights["emb_word"] = xavier(Float32, vocab, embed)
	weights["emb_world"] = xavier(Float32, onehotworld, embed)

	weights["soft_w1"] = xavier(Float32, 2*hidden, 4)
	weights["soft_w2"] = xavier(Float32, hidden, 4)
	weights["soft_w3"] = xavier(Float32, hidden, 4)
	weights["soft_b"] = zeros(Float32, 1,4)

	for k in keys(weights); weights[k] = convert(atype, weights[k]); end
	
	return weights
end

function initparams(ws; args=nothing)
	prms = Dict()
	
	for k in keys(ws); prms[k] = Adam(ws[k];lr=args["lr"]) end;

	return prms
end

# state[2k-1,2k]: hidden and cell for the k'th lstm layer
function initstate(atype, hidden, batchsize, length)
	state = Array(Any, 6)
	#forward
	state[1] = Array(Any, length+1)
	for i=1:(length+1); state[1][i] = convert(atype, zeros(batchsize, hidden)); end
	
	state[2] = Array(Any, length+1)
	for i=1:(length+1); state[2][i] = convert(atype, zeros(batchsize, hidden)); end

	#backward
	state[3] = Array(Any, length+1)
	for i=1:(length+1); state[3][i] = convert(atype, zeros(batchsize, hidden)); end
	
	state[4] = Array(Any, length+1)
	for i=1:(length+1); state[4][i] = convert(atype, zeros(batchsize, hidden)); end

	state[5] = convert(atype, zeros(batchsize, hidden*2))
	state[6] = convert(atype, zeros(batchsize, hidden*2))

	return state
end
