using Knet, AutoGrad

include("inits.jl")

function spatial(filters1, bias1, filters2, bias2, emb, x)
	c1 = relu(conv4(filters1, x; padding=0) .+ bias1)
	c2 = sigm(conv4(filters2, c1; padding=0) .+ bias2)
	h = transpose(mat(c2))
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

function lstm2(weight,bias,hidden,cell,input, encoding)
	hc = hcat(input, hidden)
	gates   = hcat(hc, encoding) * weight .+ bias
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
			x = x .* (rand!(similar(getval(x))) .> pdrops[1]) * (1/(1-pdrops[1]))
		end

		state[1], state[2] = lstm(weight1_f, bias1_f, state[1], state[2], x)


		x = words[end-i+1] * emb

		if dropout && pdrops[1] > 0.0
			x = x .* (rand!(similar(getval(x))) .> pdrops[1]) * (1/(1-pdrops[1]))
		end

		state[3], state[4] = lstm(weight1_b, bias1_b, state[3], state[4], x)
	end
end

function decode(weight1, bias1, soft_w, soft_b, state, x, mask, encoding; dropout=false, pdrops=[0.5, 0.5, 0.5])
	if dropout && pdrops[1] > 0.0
		x = x .* (rand!(similar(getval(x))) .> pdrops[1]) * (1/(1-pdrops[1]))
	end

	state[1], state[2] = lstm2(weight1, bias1, state[1], state[2], x, encoding)
	state[1] = state[1] .* mask
	state[2] = state[2] .* mask

	inp = state[1]
	if dropout && pdrops[2] > 0.0
		inp = inp .* (rand!(similar(getval(inp))) .> pdrops[2]) * (1/(1-pdrops[2]))
	end

	return (inp * soft_w .+ soft_b) .* mask
end

function loss(weights, state, words, views, ys, maskouts;lss=nothing, dropout=false, pdrops=[0.5, 0.5, 0.5])
	total = 0.0; count = 0

	#encode
	encode(weights["enc_w1_f"], weights["enc_b1_f"], weights["enc_w1_b"], weights["enc_b1_b"],
		weights["emb_word"], state, words)

	encoding = hcat(state[1], state[3])
	state[1] = hcat(state[1], state[3])
	state[2] = hcat(state[2], state[4])
	
	#decode
	for i=1:length(views)
		x = spatial(weights["filters_w1"], weights["filters_b1"], weights["filters_w2"], weights["filters_b2"], weights["emb_world"], views[i])
		ypred = decode(weights["dec_w1"], weights["dec_b1"], weights["soft_w"], weights["soft_b"], state, x, maskouts[i], encoding)
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
		state = initstate(KnetArray{Float32}, convert(Int, size(w["enc_b1_f"],2)/4), bs)

		#load data to gpu
		words = map(t->convert(KnetArray{Float32}, t), words)
		views = map(v->convert(KnetArray{Float32}, v), views)
		ys = map(t->convert(KnetArray{Float32}, t), ys)
		maskouts = map(t->convert(KnetArray{Float32}, t), maskouts)

		g = lossgradient(w, state, words, views, ys, maskouts; lss=nll, dropout=true, pdrops=args["pdrops"])
		#update weights
		for k in keys(w)
			Knet.update!(w[k], g[k], prms[k])
		end

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
		state = initstate(KnetArray{Float32}, convert(Int, size(weights["enc_b1_f"],2)/4), 1)
		encode(weights["enc_w1_f"], weights["enc_b1_f"], weights["enc_w1_b"], weights["enc_b1_b"], weights["emb_word"], state, words)

		encoding = hcat(state[1], state[3])
		state[1] = hcat(state[1], state[3])
		state[2] = hcat(state[2], state[4])

		current = instruction.path[1]
		nactions = 0
		stop = false

		println("\n$(instruction.text)")
		println("Path: $(instruction.path)")
		actions = Any[]

		while !stop
			view = state_agent_centric(maps[instruction.map], current)
			view = convert(KnetArray{Float32}, view)
			x = spatial(weights["filters_w1"], weights["filters_b1"], weights["filters_w2"], weights["filters_b2"], weights["emb_world"], view)
			ypred = decode(weights["dec_w1"], weights["dec_b1"], weights["soft_w"], weights["soft_b"], state, x, mask, encoding)
			action = indmax(Array(ypred))
			push!(actions, action)
			current = getlocation(maps[instruction.map], current, action)
			nactions += 1

			stop = nactions > args["limactions"] || action == 4 || !haskey(maps[instruction.map].nodes, (current[1], current[2]))
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
	end

	return scss / length(data)
end
function initweights(atype, hidden, vocab, embed, winit, window, onehotworld, numfilters; worldsize=[39, 39])
	weights = Dict()
	input = embed
	
	#first layer
	weights["enc_w1_f"] = xavier(Float32, input+hidden, 4*hidden)
	weights["enc_b1_f"] = zeros(Float32, 1, 4*hidden)
	weights["enc_b1_f"][1:hidden] = 1 # forget gate bias
	
	weights["enc_w1_b"] = xavier(Float32, input+hidden, 4*hidden)
	weights["enc_b1_b"] = zeros(Float32, 1, 4*hidden)
	weights["enc_b1_b"][1:hidden] = 1 # forget gate bias

	weights["dec_w1"] = xavier(Float32, input+(hidden*4), 4*hidden*2)
	weights["dec_b1"] = zeros(Float32, 1, 4*hidden*2)
	weights["dec_b1"][1:(hidden*2)] = 1 # forget gate bias

	worldfeats = (worldsize[1] - window[1] - window[2] + 2) * (worldsize[2] - window[1] - window[2] + 2) * numfilters[2]

	weights["emb_word"] = xavier(Float32, vocab, embed)
	weights["emb_world"] = xavier(Float32, worldfeats, embed)
	
	weights["filters_w1"] = xavier(Float32, window[1], window[1], onehotworld, numfilters[1])
	weights["filters_b1"] = zeros(Float32, 1, 1, numfilters[1], 1)

	weights["filters_w2"] = xavier(Float32, window[2], window[2], numfilters[1], numfilters[2])
	weights["filters_b2"] = zeros(Float32, 1, 1, numfilters[2], 1)
	
	weights["soft_w"] = xavier(Float32, hidden*2, 4)
	weights["soft_b"] = zeros(Float32, 1,4)

	for k in keys(weights); weights[k] = convert(atype, weights[k]); end
	
	return weights
end

function initparams(ws; lr=0.001)
	prms = Dict()
	
	for k in keys(ws); prms[k] = Adam(ws[k];lr=lr) end;

	return prms
end

# state[2k-1,2k]: hidden and cell for the k'th lstm layer
function initstate(atype, hidden, batchsize)
	state = Array(Any, 4)
	#forward
	state[1] = zeros(batchsize,hidden)
	state[2] = zeros(batchsize,hidden)
	
	#backward
	state[3] = zeros(batchsize,hidden)
	state[4] = zeros(batchsize,hidden)

	return map(s->convert(atype,s), state)
end


