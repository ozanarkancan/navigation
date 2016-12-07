using Knet, AutoGrad

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

function attention(words, states, att, attention_w, attention_v)
	h = hcat(words[1], hcat(states[1][2], states[3][end]))
	hu = hcat(states[6], h)
	for i=2:length(words)
		hp = hcat(words[i], hcat(states[1][i+1], states[3][end-i+1]))
		h = vcat(h, hp)
		hu = vcat(hu, hcat(states[6], hp))
	end
	
	states[5] = tanh(hu * attention_w) * attention_v

	att_s = exp(states[5])
	att_s = att_s ./ sum(att_s)

	att = att_s .* h

	return sum(att, 1), att_s
end

function decode(weight1, bias1, soft_w1, soft_w2, soft_w3, soft_b, state, x,
	mask, att; dropout=false, pdrops=[0.5, 0.5, 0.5])
	if dropout && pdrops[1] > 0.0
		x = x .* (rand!(similar(AutoGrad.getval(x))) .> pdrops[1]) * (1/(1-pdrops[1]))
	end

	state[6], state[7] = lstm2(weight1, bias1, state[6], state[7], x, att)
	state[6] = state[6] .* mask
	state[7] = state[7] .* mask

	#inp = state[6]
	if dropout && pdrops[2] > 0.0
		#inp = inp .* (rand!(similar(AutoGrad.getval(inp))) .> pdrops[2]) * (1/(1-pdrops[2]))
		state[6] = state[6] .* (rand!(similar(AutoGrad.getval(state[6]))) .> pdrops[2]) * (1/(1-pdrops[2]))
	end

	q = (state[6] * soft_w1) + x + (att * soft_w2)

	return q * soft_w3 .+ soft_b
end

function loss(weights, state, words, views, ys, maskouts, att_z;lss=nothing, dropout=false, pdrops=[0.5, 0.5, 0.5])
	total = 0.0; count = 0

	#encode
	encode(weights["enc_w1_f"], weights["enc_b1_f"], weights["enc_w1_b"], weights["enc_b1_b"],
		weights["emb_word"], state, words; dropout=dropout, pdrops=pdrops)

	state[6] = hcat(state[1][end], state[3][end])
	state[7] = hcat(state[2][end], state[4][end])
	
	#decode
	for i=1:length(views)
		x = spatial(weights["emb_world"], views[i])
		
		att,_ = attention(words, state, att_z, weights["attention_w"], weights["attention_v"])
		ypred = decode(weights["dec_w1"], weights["dec_b1"], weights["soft_w1"], weights["soft_w2"], 
			weights["soft_w3"], weights["soft_b"], state, x, maskouts[i], att;
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
		att = convert(KnetArray{Float32}, zeros(bs, convert(Int, size(words[1], 2) + 2*args["hidden"])))

		#load data to gpu
		words = map(t->convert(KnetArray{Float32}, t), words)
		views = map(v->convert(KnetArray{Float32}, v), views)
		ys = map(t->convert(KnetArray{Float32}, t), ys)
		maskouts = map(t->convert(KnetArray{Float32}, t), maskouts)

		g = lossgradient(w, state, words, views, ys, maskouts, att; lss=nll, dropout=true, pdrops=args["pdrops"])

		gclip = args["gclip"]
		if gclip > 0
			gnorm = 0
			for k in keys(g); gnorm += sumabs2(g[k]); end
			gnorm = sqrt(gnorm)

			println("Gnorm: $gnorm")

			if gnorm > gclip
				for k in keys(g)
					g[k] = g[k] * gclip / gnorm
				end
			end
		end

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
		state = initstate(KnetArray{Float32}, convert(Int, size(weights["enc_b1_f"],2)/4), 1, length(words))
		att_z = convert(KnetArray{Float32}, zeros(1, convert(Int, size(words[1], 2) + 2*size(weights["enc_b1_f"],2)/4)))
		
		encode(weights["enc_w1_f"], weights["enc_b1_f"], weights["enc_w1_b"], weights["enc_b1_b"], weights["emb_word"], state, words)
		
		state[6] = hcat(state[1][end], state[3][end])
		state[7] = hcat(state[2][end], state[4][end])
	
		current = instruction.path[1]
		nactions = 0
		stop = false

		println("\n$(instruction.text)")
		println("Path: $(instruction.path)")
		println("Filename: $(instruction.fname)")

		actions = Any[]

		while !stop
			view = state_agent_centric_multihot(maps[instruction.map], current)
			view = convert(KnetArray{Float32}, view)
			x = spatial(weights["emb_world"], view)

			att,att_s = attention(words, state, att_z, weights["attention_w"], weights["attention_v"])
			println("Attention: $(Array(att_s))")
			ypred = decode(weights["dec_w1"], weights["dec_b1"], weights["soft_w1"], weights["soft_w2"], 
			weights["soft_w3"], weights["soft_b"], state, x, mask, att)
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

	weights["dec_w1"] = xavier(Float32, input+(hidden*2) + (vocab + hidden*2), 4*hidden*2)
	weights["dec_b1"] = zeros(Float32, 1, 4*hidden*2)
	weights["dec_b1"][1:(hidden*2)] = 1 # forget gate bias

	weights["emb_word"] = xavier(Float32, vocab, embed)
	weights["emb_world"] = xavier(Float32, onehotworld, embed)

	weights["attention_w"] = xavier(Float32, hidden*2+vocab+hidden*2, hidden)
	weights["attention_v"] = xavier(Float32, hidden, 1)
	
	weights["soft_w1"] = xavier(Float32, hidden*2, hidden)
	weights["soft_w2"] = xavier(Float32, (vocab + hidden*2), hidden)
	weights["soft_w3"] = xavier(Float32, hidden, 4)
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
function initstate(atype, hidden, batchsize, length)
	state = Array(Any, 7)
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

	state[5] = convert(atype, zeros(1, length))
	
	state[6] = convert(atype, zeros(batchsize, 2*hidden))
	state[7] = convert(atype, zeros(batchsize, 2*hidden))

	return state
end


