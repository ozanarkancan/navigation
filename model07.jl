using Knet

using Base.LinAlg: axpy!

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

@knet function lstm2(x, y; fbias=1, o...)
	input  = wbf3(x, y, h; o..., f=:sigm)
	forget = wbf3(x, y, h; o..., f=:sigm, binit=Constant(fbias))
	output = wbf3(x, y, h; o..., f=:sigm)
	newmem = wbf3(x, y, h; o..., f=:tanh)
	cell = input .* newmem + cell .* forget
	h  = tanh(cell) .* output
	return h
end

@knet function encoder(xf; hidden=64, embed=64)
	remb_f = wdot(xf; out=embed)
	emb_f = drop(remb_f; pdrop=0.2)
	
	h_f = lstm(emb_f; out=hidden)

	h_f_dropped = drop(h_f; pdrop=0.5)

	h_f2 = lstm(h_f_dropped; out=hidden)

	if combine
		return wb(h_f2; out=hidden)
	end
end

#=
@knet function encoder(xf, xb; hidden=64, embed=64)
	remb_f = wdot(xf; out=embed)
	emb_f = drop(remb_f; pdrop=0.2)
	
	remb_b = wdot(xb; out=embed)
	emb_b = drop(remb_b; pdrop=0.2)

	h_f = lstm(emb_f; out=hidden)
	h_b = lstm(emb_b; out=hidden)

	h_f_dropped = drop(h_f; pdrop=0.5)
	h_b_dropped = drop(h_b; pdrop=0.5)

	h_f2 = lstm(h_f_dropped; out=hidden)
	h_b2 = lstm(h_b_dropped; out=hidden)

	if combine
		return wbf2(h_f2, h_b2; out=hidden, f=:relu)
	end
end
=#

@knet function decoder(x, embeds; dims=(39, 39, 20, 1), hidden=64, action=4)
	embedding = spatial(x; dims=dims)
	h = lstm2(embedding, embeds; out=hidden)
	return wbf(h; f=:soft, out=action)
end


#=
@knet function decoder(x, embeds; dims=(39, 39, 20, 1), hidden=64, action=4)
	embedding = spatial(x; dims=dims)
	h = lstm2(embedding, embeds; out=hidden)
	hdropped = drop(h; pdrop=0.5)
	h2 = lstm(hdropped; out=hidden)
	return wbf(h2; f=:soft, out=action)
end
=#

function train(net, data; gclip=10.0, hidden=100)
	lss = 0.0
	cnt = 0.0
	encoder, decoder = net
	for (ins, states, Y) in data
		embeds = nothing
		grads = CudaArray(Float32, hidden, 1)
		#for w=1:length(ins); embeds = sforw(encoder, ins[w], ins[end - (w-1)]; dropout=true, combine=(w == length(ins))); end
		for w=1:length(ins); embeds = sforw(encoder, ins[w]; dropout=true, combine=(w == length(ins))); end
		
		for i=1:length(states); ypred = sforw(decoder, states[i], embeds; dropout=true); lss += softloss(ypred, Y[i]); cnt += 1; end
		for i=length(Y):-1:1; _, grad = sback(decoder, Y[i], softloss; getdx=true); axpy!(1, grads, grad); end
		update!(decoder; gclip = gclip)
		
		sback(encoder, grads)
		
		for w=1:(length(ins)-1); sback(encoder); end
		update!(encoder; gclip = gclip)
		
		reset!(decoder)
		reset!(encoder)
	end

	return lss / cnt
end

function test(net, data, maps; limactions=35)
	scss = 0.0
	encoder, decoder = net

	for (instruction, wordmatrices) in data
		embeds = nothing
		#for w=1:length(wordmatrices); embeds = forw(encoder, wordmatrices[w], wordmatrices[end - (w-1)], combine=(w == length(wordmatrices))); end
		for w=1:length(wordmatrices); embeds = forw(encoder, wordmatrices[w], combine=(w == length(wordmatrices))); end

		current = instruction.path[1]
		nactions = 0
		stop = false

		println("\n$(instruction.text)")
		println("Path: $(instruction.path)")
		actions = Any[]

		while !stop
			state = state_agent_centric(maps[instruction.map], current)
			ypred = forw(decoder, state, embeds)
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

		reset!(encoder)
		reset!(decoder)
	end

	return scss / length(data)
end
