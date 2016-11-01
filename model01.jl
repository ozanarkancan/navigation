using Knet, AutoGrad

function spatial(filters, bias, emb, x)
	c = conv4(w, x) .+ bias
	h = reshape(c, 1, prod(size(c)))
	return h * emb
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

function encode(weight, bias, emb, state, words)
	x = words * emb
	for i=1:size(words, 1)
		state[1], state[2] = lstm(weight, bias, state[1], state[2], x[i, :])
	end
end

function decode(weight, bias, soft_w, soft_b, state, x)
	state[1], state[2] = lstm(weight, bias, state[1], state[2], x)
	
	return state[1] * soft_w .+ soft_b
end

function loss(weights,state,words,views,ygold)
	total = 0.0; count = 0

	#encode
	encode(weights["enc_w"], weights["enc_b"], weights["emb_word"], state, words)

	#decode
	for i=1:size(views)[end]
		x = spatial(weights["filters_w"], weights["filters_b"], weights["emb_world"], views[:, :, :, i])
		ypred = decode(weights["dec_w"], weights["dec_b"], weights["soft_w"], weights["soft_b"], state, x)
		ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
		total += sum(ygold[i, :] .* ynorm)
		#count += size(ygold,1)
		count += 1
	end
	return -total / count
end

lossgradient = grad(loss)

function initweights(atype, hidden, vocab, embed, winit, window, onehotworld, numfilters; worldsize=[39, 39])
	weights = Dict()
	input = embed
	
	weights["enc_w"] = winit*randn(input+hidden, 4*hidden)
	weights["enc_b"] = = zeros(1, 4*hidden)
	weights["enc_b"][1:hidden[k]] = 1 # forget gate bias

	weights["dec_w"] = winit*randn(input+hidden, 4*hidden)
	weights["dec_b"] = = zeros(1, 4*hidden)
	weights["dec_b"][1:hidden[k]] = 1 # forget gate bias

	worldfeats = (world[1] - window + 1) * (world[2] - window + 1) * numfilters

	param["emb_word"] = winit*randn(vocab, embed)
	param["emb_world"] = winit*randn(worldfeats, embed)
	param["filters_w"] = winit*randn(window, window, onehotworld, numfilters)
	param["filters_b"] = zeros(1, 1, numfilters, 1)
	
	weights["soft_w"] = winit*randn(hidden,4)
	weights["soft_b"] = zeros(1,4)

	for k in keys(weights)
		weights[k] = convert(atype, weights[k])
	end
	
	return weights
end

function initparams(ws; lr=0.001)
	prms = Dict()
	
	for k in keys(ws); prms[k] = Adam(ws[i];lr=lr) end;

	return prms
end

# state[2k-1,2k]: hidden and cell for the k'th lstm layer
function initstate(atype, hidden, batchsize)
	state = Array(Any, 2*length(hidden))
	for k = 1:length(hidden)
		state[2k-1] = zeros(batchsize,hidden[k])
		state[2k] = zeros(batchsize,hidden[k])
	end
	return map(s->convert(atype,s), state)
end

function train(w, prms, data; gclip=10.0)
	lss = 0.0
	cnt = 0.0
	for (words, states, Y) in data
		#load data to gpu
		words = convert(KnetArray, words)
		states = convert(KnetArray, states)
		Y = convert(KnetArray, Y)

		#call loss
		#update weights
		for k in keys(w)
			update!(w[k], g[k], prms[k])
		end
	end

	return lss / cnt
end

function test(w, data, maps; limactions=35)
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
