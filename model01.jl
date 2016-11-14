using Knet, AutoGrad

include("inits.jl")

function spatial(filters, bias, emb, x)
	c = relu(conv4(filters, x) .+ bias)
	h = transpose(mat(c))
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
	for w in words
		x = w * emb
		state[1], state[2] = lstm(weight, bias, state[1], state[2], x)
	end
end

function decode(weight, bias, soft_w, soft_b, state, x)
	state[1], state[2] = lstm(weight, bias, state[1], state[2], x)
	
	return state[1] * soft_w .+ soft_b
end

function loss(weights, state, words, views, ys, maskouts;lss=nothing)
	total = 0.0; count = 0

	#encode
	encode(weights["enc_w"], weights["enc_b"], weights["emb_word"], state, words)
	#decode
	for i=1:length(views)
		x = spatial(weights["filters_w"], weights["filters_b"], weights["emb_world"], views[i])
		ypred = decode(weights["dec_w"], weights["dec_b"], weights["soft_w"], weights["soft_b"], state, x)
		ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
		total += sum(sum(ys[i] .* ynorm, 2) .* maskouts[i])
		count += sum(maskouts[i])
	end

	nll = -total/count
	lss[1] = AutoGrad.getval(nll)
	lss[2] = AutoGrad.getval(count)
	return nll
end

lossgradient = grad(loss)

function initweights(atype, hidden, vocab, embed, winit, window, onehotworld, numfilters; worldsize=[39, 39])
	weights = Dict()
	input = embed
	
	weights["enc_w"] = xavier(Float32, input+hidden, 4*hidden)
	weights["enc_b"] = zeros(Float32, 1, 4*hidden)
	weights["enc_b"][1:hidden] = 1 # forget gate bias

	weights["dec_w"] = xavier(Float32, input+hidden, 4*hidden)
	weights["dec_b"] = zeros(Float32, 1, 4*hidden)
	weights["dec_b"][1:hidden] = 1 # forget gate bias

	worldfeats = (worldsize[1] - window[1] + 1) * (worldsize[2] - window[1] + 1) * numfilters

	weights["emb_word"] = xavier(Float32, vocab, embed)
	weights["emb_world"] = xavier(Float32, worldfeats, embed)
	weights["filters_w"] = xavier(Float32, window[1], window[1], onehotworld, numfilters[1])
	weights["filters_b"] = zeros(Float32, 1, 1, numfilters[1], 1)
	
	weights["soft_w"] = xavier(Float32, hidden, 4)
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
	state = Array(Any, 2)
	state[1] = zeros(batchsize,hidden)
	state[2] = zeros(batchsize,hidden)
	return map(s->convert(atype,s), state)
end

function train(w, prms, data; args=nothing)
	lss = 0.0
	cnt = 0.0
	nll = Float32[0, 0]
	for (words, views, ys, maskouts) in data
		bs = size(words[1], 1)
		state = initstate(KnetArray{Float32}, convert(Int, size(w["enc_b"],2)/4), bs)

		#load data to gpu
		words = map(t->convert(KnetArray{Float32}, t), words)
		views = map(v->convert(KnetArray{Float32}, v), views)
		ys = map(t->convert(KnetArray{Float32}, t), ys)
		maskouts = map(t->convert(KnetArray{Float32}, t), maskouts)

		g = lossgradient(w, state, words, views, ys, maskouts; lss=nll)
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

	for (instruction, words) in data
		words = map(v->convert(KnetArray{Float32},v), words)
		state = initstate(KnetArray{Float32}, convert(Int, size(weights["enc_b"],2)/4), 1)
		encode(weights["enc_w"], weights["enc_b"], weights["emb_word"], state, words)

		current = instruction.path[1]
		nactions = 0
		stop = false

		println("\n$(instruction.text)")
		println("Path: $(instruction.path)")
		actions = Any[]

		while !stop
			view = state_agent_centric(maps[instruction.map], current)
			view = convert(KnetArray{Float32}, view)
			x = spatial(weights["filters_w"], weights["filters_b"], weights["emb_world"], view)
			ypred = decode(weights["dec_w"], weights["dec_b"], weights["soft_w"], weights["soft_b"], state, x)
			action = indmax(Array(ypred))
			push!(actions, action)
			current = getlocation(maps[instruction.map], current, action)
			nactions += 1

			stop = nactions > args["limnactions"] || action == 4 || !haskey(maps[instruction.map].nodes, (current[1], current[2]))
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
