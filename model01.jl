using Knet, AutoGrad

function spatial(w, x)
	return conv4(w, x)
end

function lstm(weight,bias,hidden,cell,input;encode=true)
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

function predict(w, s, x; encode=true)
	if encode
		x = x * w[end-3]
	else
		x = spatial(w[end-2], x)
	end

	for i = 1:2:length(s)
		(s[i],s[i+1]) = lstm(w[i+1],w[i+1],s[i],s[i+1],x;encode=encode)
		x = s[i]
	end
	return x * w[end-1] .+ w[end]
end

function loss(param,state,sequence,range=1:length(sequence)-1)
	total = 0.0; count = 0
	atype = typeof(getval(param[1]))
	input = convert(atype,sequence[first(range)])
	for t in range
		ypred = predict(param,state,input)
		ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
		ygold = convert(atype,sequence[t+1])
		total += sum(ygold .* ynorm)
		count += size(ygold,1)
		input = ygold
	end
	return -total / count
end

lossgradient = grad(loss)

# param[2k-1,2k]: weight and bias for the k'th lstm layer
# param[end-2]: embedding matrix
# param[end-1,end]: weight and bias for final prediction
function initweights(atype, hidden, vocab, embed, winit, window, onehot)
	param = Array(Any, 2*length(hidden)+3)
	input = embed
	for k = 1:length(hidden)
		param[2k-1] = winit*randn(input+hidden[k], 4*hidden[k])
		param[2k]   = zeros(1, 4*hidden[k])
		param[2k][1:hidden[k]] = 1 # forget gate bias
		input = hidden[k]
	end
	param[end-3] = winit*randn(vocab,embed)
	param[end-2] = winit*randn(window, window, onehot, 1)
	param[end-1] = winit*randn(hidden[end],vocab)
	param[end] = zeros(1,vocab)
	return map(p->convert(atype,p), param)
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
	for (ins, states, Y) in data
		#load data to gpu
		ins = convert(KnetArray, ins)
		states = convert(KnetArray, states)
		Y = convert(KnetArray, Y)
		for w=1:length(ins); sforw(net, ins[w]; dropout=true); end
		for i=1:length(states); ypred = sforw(net, states[i]; decode=true, dropout=true); lss += softloss(ypred, Y[i]); cnt += 1; end
		for i=length(Y):-1:1; sback(net, Y[i], softloss); end
		for w in ins; sback(net); end

		update!(net; gclip = gclip)
		reset!(net)
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
