using Knet

@knet function spatial(x; dims=(39, 39, 25, 1))
	w = par(init=Gaussian(0, 0.001), dims=dims)
	c = conv(w, x)
	return c
end

@knet function wbf3(x1, x2, x3; f=:sigm, o...)
	y1 = wdot(x1; o...)
	y2 = wdot(x2; o...)
	y3 = wdot(x3; o...)
	x4 = add(y2,y1)
	x5 = add(y3, x4)
	y4 = bias(x5; o...)
	return f(y4; o...)
end

@knet function wb2(x1, x2; o...)
	y1 = wdot(x1; o...)
	y2 = wdot(x2; o...)
	x3 = add(y2,y1)
	y3 = bias(x3; o...)
	return y3
end

@knet function model(x; dims=(21, 39, 25, 1), hidden=128, embed=128, actions=4)
	if !decode
		rembedding = wdot(x; out=embed)
		embedding = drop(rembedding; pdrop=0.2)
		input  = wbf2(embedding, h; out=hidden, f=:sigm)
	        forget = wbf2(embedding, h; out=hidden, f=:sigm)
	        output = wbf2(embedding, h; out=hidden, f=:sigm)
	        newmem = wbf2(embedding, h; out=hidden, f=:tanh)
	else
		rembedding = spatial(x; dims=dims)
		embedding = drop(rembedding; pdrop=0.2)

		input  = wbf2(embedding, h; out=hidden, f=:sigm)
	        forget = wbf2(embedding, h; out=hidden, f=:sigm)
	        output = wbf2(embedding, h; out=hidden, f=:sigm)
	        newmem = wbf2(embedding, h; out=hidden, f=:tanh)
	end

	cell = input .* newmem + cell .* forget
	h  = tanh(cell) .* output

	hdrop = drop(h; pdrop=0.5)

	if !decode
		input2  = wbf2(hdrop, h2; out=hidden, f=:sigm)
	        forget2 = wbf2(hdrop, h2; out=hidden, f=:sigm)
	        output2 = wbf2(hdrop, h2; out=hidden, f=:sigm)
	        newmem2 = wbf2(hdrop, h2; out=hidden, f=:tanh)
	else
		input2  = wbf2(hdrop, h2; out=hidden, f=:sigm)
	        forget2 = wbf2(hdrop, h2; out=hidden, f=:sigm)
	        output2 = wbf2(hdrop, h2; out=hidden, f=:sigm)
	        newmem2 = wbf2(hdrop, h2; out=hidden, f=:tanh)
	end
	
	cell2 = input2 .* newmem2 + cell2 .* forget2
	h2  = tanh(cell2) .* output2
	
	if decode
		return wbf(h2; out=actions, f=:soft)
	end
end

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
