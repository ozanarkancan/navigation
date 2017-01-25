using Knet, AutoGrad, Logging

include("inits.jl")

function spatial(filters1, bias1, filters2, bias2, filters3, bias3, emb, x)
	c1 = relu(conv4(filters1, x; padding=0) .+ bias1)
	c2 = relu(conv4(filters2, c1; padding=0) .+ bias2)
	c3 = sigm(conv4(filters3, c2; padding=0) .+ bias3)
	h = transpose(mat(c3))
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

function loss(weights, state, words, views, ys, maskouts;lss=nothing, dropout=false, pdrops=[0.5, 0.5, 0.5], rewards=nothing)
	total = 0.0; count = 0

	#encode
	encode(weights["enc_w1_f"], weights["enc_b1_f"], weights["enc_w1_b"], weights["enc_b1_b"],
		weights["emb_word"], state, words; dropout=dropout, pdrops=pdrops)

	state[5] = hcat(state[1][end], state[3][end])
	state[6] = hcat(state[2][end], state[4][end])
	
	#decode
	for i=1:length(views)
		x = spatial(weights["filters_w1"], weights["filters_b1"], weights["filters_w2"], weights["filters_b2"], 
			weights["filters_w3"], weights["filters_b3"], weights["emb_world"], views[i])
		
		ypred = decode(weights["dec_w1"], weights["dec_b1"], weights["soft_w1"], weights["soft_w2"], 
			weights["soft_w3"], weights["soft_b"], state, x, maskouts[i];
			dropout=dropout, pdrops=pdrops)

		ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
		if rewards == nothing
			total += sum((ys[i] .* ynorm) .* maskouts[i])
		else
			total += sum((ys[i] .* ynorm) .* maskouts[i]) * rewards[i]
		end
		count += sum(maskouts[i])
	end

	nll = -total/count
	lss[1] = AutoGrad.getval(nll)
	lss[2] = AutoGrad.getval(count)
	return nll
end


function predictV(X, V1, b1, V2, b2)
	h = relu(X * V1 .+ b1)
	return h * V2 .+ b2
end

mse(w, X, Y) = (sum(abs2(Y-predictV(X, w["V1"], w["V1b"], w["V2"], w["V2b"]))) / size(X,1))

lossgradient = grad(loss)
lossgradientV = grad(mse)


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

function train_pg(weights, prms, data, maps; args=nothing)
	total = 0.0
	mask = convert(KnetArray, ones(Float32, 1,1))
	counter = 0
	grads = Dict()
	for (instruction, words) in data
		counter += 1
		words = map(v->convert(KnetArray{Float32},v), words)
		views = Any[]
		actions = Any[]
		rewards = Any[]
		masks = Any[]

		state = initstate(KnetArray{Float32}, convert(Int, size(weights["enc_b1_f"],2)/4), 1, length(words))

		encode(weights["enc_w1_f"], weights["enc_b1_f"], weights["enc_w1_b"], weights["enc_b1_b"],
			weights["emb_word"], state, words)

		state[5] = hcat(state[1][end], state[3][end])
		state[6] = hcat(state[2][end], state[4][end])

		Xs = nothing

		current = instruction.path[1]
		nactions = 0
		stop = false

		actions = Any[]
		info("Path: $(instruction.path)")

		while !stop
			view = state_agent_centric(maps[instruction.map], current)
			view = convert(KnetArray{Float32}, view)
			push!(views, view)

			x = spatial(weights["filters_w1"], weights["filters_b1"], weights["filters_w2"], weights["filters_b2"], 
				weights["filters_w3"], weights["filters_b3"], weights["emb_world"], view)
			
			ypred = decode(weights["dec_w1"], weights["dec_b1"], weights["soft_w1"], weights["soft_w2"], 
			weights["soft_w3"], weights["soft_b"], state, x, mask)

			Xs = Xs == nothing ? state[5] : vcat(Xs, state[5])

			a = sample(Array(ypred))
			info("Sampled: $a")
			action = zeros(Float32, 1, 4)
			action[1, a] = 1.0

			push!(actions, convert(KnetArray, action))
			current = getlocation(maps[instruction.map], current, a)
			nactions += 1

			if nactions > args["limactions"] || !haskey(maps[instruction.map].nodes, (current[1], current[2]))
				stop = true
				push!(rewards, -1.0)
			elseif a == 4
				stop = true
				if current == instruction.path[end]
					push!(rewards, length(instruction.path)*1.0)
				else
					push!(rewards, -1.0)
					#=
				x1,y1,z1 = instruction.path[end]
				x2,y2,z2 = current
				dist = norm([x1 y1] - [x2 y2])
				push!(rewards, -1.0 * dist)=#
				end
			else
				if current in instruction.path
					push!(rewards, -1.0/length(instruction.path))
					#push!(rewards, 1.0/length(instruction.path))
				else
					push!(rewards, -1.0)
					#push!(rewards, -1.0/length(instruction.path))
				end
			end
			info("Reward: $(rewards[end])")
		end

		V = predictV(Xs, weights["V1"], weights["V1b"], weights["V2"], weights["V2b"])
		v = Array(V)
		info("PredV: $v")
		disc_rewards = discount(rewards; γ=args["gamma"])
		total += disc_rewards[1]
		rs = Any[]
		delta = zeros(Float32, length(rewards), 1)
		#total += sum(rewards)

		#for r in rewards
		for r=0:(length(disc_rewards)-1)
			#push!(masks, convert(KnetArray, (0.9^r) * disc_rewards[r+1] * ones(Float32, 1,1)))
			push!(masks, convert(KnetArray, ones(Float32, 1,1)))
			delta[r+1, 1] = disc_rewards[r+1] - v[r+1]
			push!(rs, (args["gamma"]^r) * delta[r+1, 1])
		end

		state = initstate(KnetArray{Float32}, convert(Int, size(weights["enc_b1_f"],2)/4), 1, length(words))
		nll = Float32[0, 0]
		g = lossgradient(weights, state, words, views, actions, masks; lss=nll, dropout=false, pdrops=args["pdrops"], rewards=rs)

		gV = lossgradientV(weights, Xs, convert(KnetArray, disc_rewards))

		for k in keys(g)
			grads[k] = haskey(grads, k) ? grads[k] + g[k] : g[k]
		end

		for k in keys(gV)
			grads[k] = haskey(grads, k) ? grads[k] + gV[k] : gV[k]
		end

		if counter % 10 == 0 || counter == length(data)
			gnorm = 0

			for k in keys(grads); gnorm += sumabs2(grads[k]); end
			gnorm = sqrt(gnorm)
			gclip = args["gclip"]

			debug("Gnorm: $gnorm")

			if gnorm > gclip
				for k in keys(grads)
					grads[k] = grads[k] * gclip / gnorm
				end
			end

			for k in keys(grads)
				Knet.update!(weights[k], grads[k] / (counter % 10 == 0 ? 10 : counter), prms[k])
			end

			grads = Dict()
		end
	end
	return total / length(data)
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

		loss(w, state, words, views, ys, maskouts; lss=nll, dropout=false, pdrops=args["pdrops"])

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
			view = state_agent_centric(maps[instruction.map], current)
			view = convert(KnetArray{Float32}, view)
			x = spatial(weights["filters_w1"], weights["filters_b1"], weights["filters_w2"], weights["filters_b2"], 
				weights["filters_w3"], weights["filters_b3"], weights["emb_world"], view)

			ypred = decode(weights["dec_w1"], weights["dec_b1"], weights["soft_w1"], weights["soft_w2"], 
			weights["soft_w3"], weights["soft_b"], state, x, mask)
			action = 0
			if args["greedy"]
				action = indmax(Array(ypred))
			else
				action = sample(Array(ypred))
			end
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
				view = state_agent_centric(maps[instruction.map], current)
				view = convert(KnetArray{Float32}, view)
				x = spatial(weights["filters_w1"], weights["filters_b1"], weights["filters_w2"], weights["filters_b2"], 
					weights["filters_w3"], weights["filters_b3"], weights["emb_world"], view)

				ypred = decode(weights["dec_w1"], weights["dec_b1"], weights["soft_w1"], weights["soft_w2"], 
				weights["soft_w3"], weights["soft_b"], state, x, mask)
				
				action = 0
				if args["greedy"]
					action = indmax(Array(ypred))
				else
					action = sample(Array(ypred))
				end

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


function initweights(atype, hidden, vocab, embed, window, onehotworld, numfilters; worldsize=[39, 39])
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

	worldfeats = (worldsize[1] - window[1] - window[2] - window[3] + 3) * (worldsize[2] - window[1] - window[2] - window[3] + 3) * numfilters[3]

	weights["emb_word"] = xavier(Float32, vocab, embed)
	weights["emb_world"] = xavier(Float32, worldfeats, embed)
		
	weights["filters_w1"] = xavier(Float32, window[1], window[1], onehotworld, numfilters[1])
	weights["filters_b1"] = zeros(Float32, 1, 1, numfilters[1], 1)

	weights["filters_w2"] = xavier(Float32, window[2], window[2], numfilters[1], numfilters[2])
	weights["filters_b2"] = zeros(Float32, 1, 1, numfilters[2], 1)

	weights["filters_w3"] = xavier(Float32, window[3], window[3], numfilters[2], numfilters[3])
	weights["filters_b3"] = zeros(Float32, 1, 1, numfilters[3], 1)

	weights["attention_w"] = xavier(Float32, hidden*2+hidden*2, hidden)
	weights["attention_v"] = xavier(Float32, hidden, 1)

	weights["soft_w1"] = xavier(Float32, 2*hidden, 4)
	weights["soft_w2"] = xavier(Float32, hidden, 4)
	weights["soft_w3"] = xavier(Float32, hidden, 4)
	weights["soft_b"] = zeros(Float32, 1,4)

	weights["V1"] = xavier(Float32, 2*hidden, hidden)
	weights["V1b"] = xavier(Float32, 1, hidden)
	weights["V2"] = xavier(Float32, hidden, 1)
	weights["V2b"] = xavier(Float32, 1, 1)

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
