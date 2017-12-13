using Knet, AutoGrad, Logging

include("inits.jl")

function initweights(atype, hidden, vocab, embed, window, onehotworld, numfilters; worldsize=[5, 20], args=nothing, premb=nothing, winit=1.0)
    weights = Dict()
    input = premb == nothing ? embed : size(premb, 2)

    #first layer
    weights["enc_w_f"] = xavier(Float32, input+hidden, 4*hidden)
    weights["enc_b_f"] = zeros(Float32, 1, 4*hidden)
    weights["enc_b_f"][1:hidden] = 1 # forget gate bias

    weights["enc_w_b"] = xavier(Float32, input+hidden, 4*hidden)
    weights["enc_b_b"] = zeros(Float32, 1, 4*hidden)
    weights["enc_b_b"][1:hidden] = 1 # forget gate bias

    #vocab 300 for embeddings, 512 for standard
    weights["emb_word"] = premb == nothing ? xavier(Float32, vocab, embed) : premb
    
    #decoder
    if args["percp"] && args["encoding"] == "grid"
        if args["worldatt"] != 0 && length(numfilters) == 1
            win1, win2 = window[1]
            worldfeats = (worldsize[1] - win1 + 1) * (worldsize[2] - win2 + 1)
        else
            sw1 = sum(map(x->x[1], window))
            sw2 = sum(map(x->x[2], window))
            worldfeats = (worldsize[1] - sw1 + length(window)) * (worldsize[2] - sw2 + length(window)) * numfilters[end]
        end
        weights["emb_world"] = xavier(Float32, worldfeats, embed)

        fs = Any[]
        bs = Any[]

        for i=1:length(numfilters)
            inpch = i == 1 ? onehotworld : numfilters[i-1]
            if args["worldatt"] != 0 && i == 2
                inpch = 1
            end

            win1, win2 = window[i]
            push!(fs, xavier(Float32, win1, win2, inpch, numfilters[i]) * winit)
            push!(bs, zeros(Float32, 1, 1, numfilters[i], 1))
        end

        if args["worldatt"] != 0
            wattin = args["attinwatt"] != 0 ? 4*hidden : 2*hidden
            weights["wa1"] = xavier(Float32, wattin, args["worldatt"])
            weights["wa2"] = xavier(Float32, args["worldatt"], numfilters[1])
        end
        weights["filters_w"] = fs
        weights["filters_b"] = bs
    elseif args["percp"]
        weights["emb_world"] = xavier(Float32, onehotworld, embed)
    end
    
    if !args["percp"] && args["preva"] && !args["att"]
        weights["emb_world"] = xavier(Float32, 4, embed)
        weights["dec_w"] = xavier(Float32, embed + hidden*2, 4*hidden*2)
    elseif !args["percp"] && args["att"] && args["preva"]
        weights["emb_world"] = xavier(Float32, 4, embed)
        weights["dec_w"] = xavier(Float32, embed + hidden*2 + hidden*2, 4*hidden*2)
    elseif args["percp"] && args["att"] && args["preva"]
        weights["dec_w"] = xavier(Float32, embed + 4 + hidden*2 + hidden*2, 4*hidden*2)
    elseif args["percp"] && !args["att"] && args["preva"]
        weights["dec_w"] = xavier(Float32, embed + 4 + hidden*2, 4*hidden*2)
    elseif args["percp"] && args["att"] && !args["preva"]
        weights["dec_w"] = xavier(Float32, embed + hidden*2 + hidden*2, 4*hidden*2)
    elseif args["percp"] && !args["att"] && !args["preva"]
        weights["dec_w"] = xavier(Float32, embed + hidden*2, 4*hidden*2)
    end
    
    weights["dec_b"] = zeros(Float32, 1, 4*hidden*2)
    weights["dec_b"][1:hidden*2] = 1 # forget gate bias
 
    #attention
    if args["att"] || args["attinwatt"] != 0
        atthidden = args["attinwatt"] != 0 ? args["attinwatt"] : hidden
        #fenc, benc, dechid
        weights["attention_w"] = xavier(Float32, hidden*2+hidden*2, atthidden)
        weights["attention_v"] = xavier(Float32, atthidden, 1)
    end

    #output
    weights["soft_h"] = xavier(Float32, 2*hidden, 4)

    if args["inpout"]
        weights["soft_inp"] = xavier(Float32, embed, 4)
    end

    if args["att"] && args["attout"]
        weights["soft_att"] = xavier(Float32, hidden*2, 4)
    end

    if args["preva"] && args["prevaout"]
        weights["soft_preva"] = xavier(Float32, 4, 4)
    end

    weights["soft_b"] = zeros(Float32, 1,4)

    for k in keys(weights)
        if startswith(k, "filter")
            ws = map(t->convert(atype, t), weights[k])
            weights[k] = ws
        else
            weights[k] = convert(atype, weights[k])
        end
    end

    return weights
end

function cnn(filters, bias, x)
    inp = x
    for i=1:length(filters)-1
        inp = relu.(conv4(filters[i], inp; padding=0) .+ bias[i])
    end
    inp = sigm.(conv4(filters[end], inp; padding=0) .+ bias[end])
    return transpose(mat(inp))
end

function cnn(filters, bias, worldatt, x)
    inp = x
    for i=1:length(filters)-1
        inp = relu.(conv4(filters[i], inp; padding=0) .+ bias[i])
        if i == 1
            #inp = conv4(worldatt, inp; padding=0)
            inp = relu.(conv4(worldatt, inp; padding=0))
        end
    end
    inp = sigm.(conv4(filters[end], inp; padding=0) .+ bias[end])
    if length(filters) == 1
        #inp = conv4(worldatt, inp; padding=0)
        inp = sigm.(conv4(worldatt, inp; padding=0))
    end
    return transpose(mat(inp))
end

function spatial(emb, x) #multihot
    return x * emb
end

function spatial(filters, bias, emb, x) #grid
    h = cnn(filters, bias, x)
    return h * emb
end

function spatial(filters, bias, emb, worldatt, x) # world att
    h = cnn(filters, bias, worldatt, x)
    return h * emb
end

function lstm(weight,bias,hidden,cell,input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm.(gates[:,1:hsize])
    ingate  = sigm.(gates[:,1+hsize:2hsize])
    outgate = sigm.(gates[:,1+2hsize:3hsize])
    change  = tanh.(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh.(cell)
    return (hidden,cell)
end

function attention(states, attention_w, attention_v)
    h = hcat(states[1][2], states[3][end])
    hu = hcat(states[5], h)
    for i=2:(length(states[1])-1)
        hp = hcat(states[1][i+1], states[3][end-i+1])
        h = vcat(h, hp)
        hu = vcat(hu, hcat(states[5], hp))
    end

    raw_att = tanh.(hu * attention_w) * attention_v
    raw_att = raw_att .- maximum(raw_att)

    att_s = exp(raw_att)
    att_s = att_s ./ sum(att_s)

    att = att_s .* h

    return sum(att, 1), att_s
end

function worldattention(prevh, wa1, wa2)
    h = tanh.(prevh * wa1) * wa2
    h = h .- maximum(h)
    att_p = exp.(h)
    att_p = att_p ./ sum(att_p)
    return reshape(att_p, 1, 1, size(wa2, 2), 1)
end

#dropout before the loop might accelerate the execution
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

#x might be view or preva
function decode(weight, bias, soft_h, soft_b, state, x; soft_inp=nothing, soft_att=nothing, 
    soft_preva=nothing, preva=nothing, att=nothing, dropout=false, pdrops=[0.5, 0.5], prevainp=false)
    
    inp = x
    
    if preva != nothing && prevainp
        inp = hcat(inp, preva)
    end

    if att != nothing
        inp = hcat(inp, att)
    end

    if dropout && pdrops[1] > 0.0
        inp = inp .* (rand!(similar(AutoGrad.getval(inp))) .> pdrops[1]) * (1/(1-pdrops[1]))
    end

    state[5], state[6] = lstm(weight, bias, state[5], state[6], inp)

    inph = state[5]
    if dropout && pdrops[2] > 0.0
        inph = inph .* (rand!(similar(AutoGrad.getval(inph))) .> pdrops[2]) * (1/(1-pdrops[2]))
    end

    q = (inph * soft_h) .+ soft_b
    
    if soft_inp != nothing
        q = q + inp[:, 1:size(x,2)] * soft_inp
    end
    
    if soft_preva != nothing
        q = q + preva * soft_preva
    end

    if soft_att != nothing
        q = q + att * soft_att
    end

    return q
end

function probs(linear)
    linear = linear .- maximum(linear, 2)
    ps = exp.(linear) ./ sum(exp.(linear), 2)
    return ps
end

function sample(ps)
    c_probs = cumsum(ps, 2)
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

function loss(w, state, words, ys; lss=nothing, views=nothing, as=nothing, dropout=false, encpdrops=[0.5, 0.5], decpdrops=[0.5, 0.5], args=nothing)
    total = 0.0; count = 0

    #encode
    encode(w["enc_w_f"], w["enc_b_f"], w["enc_w_b"], w["enc_b_b"], w["emb_word"],
        state, words; dropout=dropout, pdrops=encpdrops)

    state[5] = hcat(state[1][end], state[3][end])
    state[6] = hcat(state[2][end], state[4][end])

    #decode
    for i=1:length(ys)
        att,_ = args["att"] || args["attinwatt"] != 0 ? attention(state, w["attention_w"], w["attention_v"]) : (nothing, nothing)
        
        if !args["percp"]
            x = spatial(w["emb_world"], as[i])
        elseif args["encoding"] == "grid" && args["worldatt"] != 0
            wattinp = args["attinwatt"] == 0 ? state[5] : hcat(state[5], att)
            worldatt = worldattention(wattinp, w["wa1"], w["wa2"])
            x =  spatial(w["filters_w"], w["filters_b"], w["emb_world"], worldatt, views[i]) # world att
        elseif args["encoding"] == "grid" 
            x = spatial(w["filters_w"], w["filters_b"], w["emb_world"], views[i])
        else
            x = spatial(w["emb_world"], views[i])
        end
        
        soft_inp = args["inpout"] ? w["soft_inp"] : nothing
        soft_att = args["attout"] ? w["soft_att"] : nothing
        soft_preva = args["prevaout"] ? w["soft_preva"] : nothing
        preva = !args["preva"] ? nothing : as[i]
        prevainp = args["preva"] && args["percp"]
        
        att = args["att"] ? att : nothing

        ypred = decode(w["dec_w"], w["dec_b"], w["soft_h"], w["soft_b"], state, x; soft_inp=soft_inp, soft_att=soft_att, 
            soft_preva=soft_preva, preva=preva, att=att, dropout=dropout, pdrops=decpdrops, prevainp=prevainp)

        ynorm = logp(ypred,2)
        total += sum(ys[i] .* ynorm)
        
        count += 1
    end

    nll = -total/count
    lss[1] = AutoGrad.getval(nll)
    lss[2] = AutoGrad.getval(count)
    return nll
end

function loss_global(w, words, goldactions, instruction, maps, state; dropout=false, encpdrops=[0.5, 0.5], decpdrops=[0.5, 0.5], args=nothing, lss=nothing)
    nll = 0.0
    beamsize = args["beamsize"]
    
    encode(w["enc_w_f"], w["enc_b_f"], w["enc_w_b"], w["enc_b_b"], w["emb_word"], state, words; dropout=dropout, pdrops=encpdrops)

    state[5] = hcat(state[1][end], state[3][end])
    state[6] = hcat(state[2][end], state[4][end])
        
    cstate5 = copy(state[5])
    cstate6 = copy(state[6])
        
    current = instruction.path[1]
    #1: score
    #2: prev hidden
    #3: prev cell
    #4: current location
    #5: depth
    #6: stopped
    #7: isgold
    #8: prev actions
    cands = Array{Tuple{Float64, Any, Any, Any, Any, Any, Any, Any}, 1}()
    push!(cands, (0.0, cstate5, cstate6, current, 0, false, true, Any[4]))

    nactions = 0
    stop = false
    stopsearch = false
    araw = args["preva"] ? reshape(Float32[0.0, 0.0, 0.0, 1.0], 1, 4) : nothing
    goldindex = 0

    while !stopsearch
        newcands = Array{Tuple{Any, Any, Any, Any, Any, Any, Any, Any}, 1}()
        newcand = false
        for cand in cands
            current = cand[4]
            state[5] = cand[2]
            state[6] = cand[3]
            depth = cand[5]
            prevActions = cand[end]
            lastAction = prevActions[end]
            stopped = cand[6]
            isgold = cand[7]

            if araw != nothing
                araw[:] = 0.0
                araw[1, lastAction] = 1.0
            end

            if stopped
                push!(newcands, cand)
                continue
            end

            view = !args["percp"] ? nothing : args["encoding"] == "grid" ? 
            state_agent_centric(maps[instruction.map], current) : state_agent_centric_multihot(maps[instruction.map], current)
            view = args["percp"] ? convert(KnetArray{Float32}, view) : nothing
            preva = araw != nothing ? convert(KnetArray{Float32}, araw)  : araw

            if !args["percp"]
                x = spatial(w["emb_world"], preva)
            elseif args["encoding"] == "grid" && args["worldatt"] != 0
                worldatt = worldattention(state[5], w["wa1"], w["wa2"])
                x =  spatial(w["filters_w"], w["filters_b"], w["emb_world"], worldatt, view) # world att
            elseif args["encoding"] == "grid" 
                x = spatial(w["filters_w"], w["filters_b"], w["emb_world"], view)
            else
                x = spatial(w["emb_world"], view)
            end

            soft_inp = args["inpout"] ? w["soft_inp"] : nothing
            soft_att = args["attout"] ? w["soft_att"] : nothing
            soft_preva = args["prevaout"] ? w["soft_preva"] : nothing
            preva = !args["preva"] ? nothing : preva
            prevainp = args["preva"] && args["percp"]

            att,_ = args["att"] ? attention(state, w["attention_w"], w["attention_v"]) : (nothing, nothing)
            ypred = decode(w["dec_w"], w["dec_b"], w["soft_h"], w["soft_b"], state, x; soft_inp=soft_inp, soft_att=soft_att, 
            soft_preva=soft_preva, preva=preva, att=att, dropout=dropout, pdrops=decpdrops, prevainp=prevainp)

            for i=1:4
                myisgold = isgold
                mystop = stopped
                nactions = depth + 1
                actcopy = copy(prevActions)
                
                if isgold
                    myisgold = i == goldactions[nactions]
                end

                if nactions >= length(goldactions) && i < 4
                    mystop = true
                end
                push!(actcopy, i)
                cur = identity(current)

                if i < 4
                    cur = getlocation(maps[instruction.map], cur, i)
                end

                if (i == 1 && !haskey(maps[instruction.map].edges[(current[1], current[2])], (cur[1], cur[2]))) || i==4
                    mystop = true
                end

                newscore = cand[1] + ypred[1, i]

                push!(newcands, (newscore, state[5], state[6], cur, nactions, mystop, myisgold, actcopy))
                newcand = true
            end
        end
        stopsearch = !newcand
        if newcand
            sort!(newcands; by=x->x[1], rev=true)
            l = length(newcands) < beamsize ? length(newcands) : beamsize
            goldindex = findfirst(x->x[7], newcands)
            cands = newcands[1:l]
            debug("l: $l")
            for index=1:l
                debug("Beam #$index : $(AutoGrad.getval(newcands[index][1])) - $(newcands[index][end])")
            end

            if goldindex > beamsize || newcands[goldindex][5] == length(goldactions)
                stopsearch = true
                glob = vcat(first(cands[1]))
                for ind=2:l
                    glob = vcat(glob, first(cands[ind]))
                end

                if goldindex > beamsize
                    debug("gind: $goldindex")
                    glob = vcat(glob, first(newcands[goldindex]))
                    ynorm = logp(glob)
                    debug("ynorm: $(Array(AutoGrad.getval(ynorm)))")
                    nll = -ynorm[end]
                else
                    debug("gind: $goldindex")
                    ynorm = logp(glob)
                    debug("ynorm: $(Array(AutoGrad.getval(ynorm)))")
                    nll = -ynorm[goldindex]
                end

                debug("nll: $(AutoGrad.getval(nll))")
                debug("gold: $(AutoGrad.getval(newcands[goldindex][1]))")
                debug("goldactions: $(goldactions)")
                #=
                debug("glob: $(AutoGrad.getval(glob))")
                z = log(sum(exp(glob)))
                debug("z: $(AutoGrad.getval(z))")
                nll = -newcands[goldindex][1] + z
                debug("gold: $(AutoGrad.getval(newcands[goldindex][1]))")
                debug("nll: $(AutoGrad.getval(nll))")
                debug("goldactions: $(goldactions)")
                debug("beam-gold: $(newcands[goldindex][end])")
                for index=1:l
                    debug("Beam #$index : $(newcands[index][end])")
                end
                =#
            end
        end
    end
    debug("Exiting nll: $(AutoGrad.getval(nll))")
    lss[1] = AutoGrad.getval(nll)
    return nll
end

lossgradient = grad(loss)
lossglobalgradient = grad(loss_global)

function clip_grad(g; args=args)
    gclip = args["gclip"]
    if gclip > 0
        gnorm = 0
        for k in keys(g)
            if startswith(k, "filter")
                for el in g[k]
                    gnorm += sum(abs2, el)
                end
            else
                gnorm += sum(abs2, g[k])
            end
        end
        gnorm = sqrt(gnorm)

        #debug("Gnorm: $gnorm")

        if gnorm > gclip
            for k in keys(g)
                if startswith(k, "filter")
                    gs = Any[]
                    for el in g[k]
                        push!(gs, el * gclip / gnorm)
                    end
                    g[k] = gs
                else
                    g[k] = g[k] * gclip / gnorm
                end
            end
        end
    end
end

function updatew(w,g,prms)
    #update weights
    for k in keys(g)
        if startswith(k, "filter")
            for ind=1:length(g[k])
                Knet.update!(w[k][ind], g[k][ind], prms[k][ind])
            end
        else
            Knet.update!(w[k], g[k], prms[k])
        end
    end
end

function train(w, prms, data; args=nothing, updatelimit=0)
    lss = 0.0
    cnt = 0.0
    nll = Float32[0, 0]
    for (words, views, ys, _) in data
        bs = size(words[1], 1)
        state = initstate(KnetArray{Float32}, args["hidden"], bs, length(words))
        
        #load data to gpu
        words = map(t->convert(KnetArray{Float32}, t), words)
        ys = map(t->convert(KnetArray{Float32}, t), ys)
        
        views = args["percp"] ? map(v->convert(KnetArray{Float32}, v), views) : nothing
        acts = nothing
        
        if args["preva"]
            as = Any[]
            push!(as, reshape(Float32[0.0, 0.0, 0.0, 1.0], 1, 4))
            append!(as, ys[1:(end-1)])
            acts = args["preva"] ? map(t->convert(KnetArray{Float32}, t), as) : nothing
        end

        g = lossgradient(w, state, words, ys; lss=nll, views=views, as=acts, dropout=true, encpdrops=args["encdrops"], decpdrops=args["decdrops"], args=args)
        clip_grad(g;args=args)
        updatew(w,g,prms)
        
        lss += nll[1] * nll[2]
        cnt += nll[2]

        if updatelimit != 0 && cnt >= updatelimit
            break
        end
    end
    return lss / cnt
end

function train_global(w, prms, data, maps; args=nothing)
    lss = 0.0
    cnt = 0.0
    nll = Float64[0, 0]

    for (instruction, words) in data
        words = map(t->convert(KnetArray{Float32}, t), words)
        goldactions = getactions(instruction.path)
        bs = size(words[1], 1)
        state = initstate(KnetArray{Float32}, args["hidden"], bs, length(words))

        g = lossglobalgradient(w, words, goldactions, instruction, maps, state; dropout=true, encpdrops=args["encdrops"], decpdrops=args["decdrops"], args=args, lss=nll)
        clip_grad(g;args=args)
        updatew(w,g,prms)

        lss += nll[1]
        cnt += 1.0
    end

    return lss / cnt
end


function train_loss(w, data; args=nothing)
    lss = 0.0
    cnt = 0.0
    nll = Float32[0, 0]
    for (words, views, ys, _) in data
        bs = size(words[1], 1)
        state = initstate(KnetArray{Float32}, args["hidden"], bs, length(words))
        
        #load data to gpu
        words = map(t->convert(KnetArray{Float32}, t), words)
        ys = map(t->convert(KnetArray{Float32}, t), ys)
        
        views = args["percp"] ? map(v->convert(KnetArray{Float32}, v), views) : nothing
        acts = nothing
        
        if args["preva"]
            as = Any[]
            push!(as, reshape(Float32[0.0, 0.0, 0.0, 1.0], 1, 4))
            append!(as, ys[1:(end-1)])
            acts = args["preva"] ? map(t->convert(KnetArray{Float32}, t), as) : nothing
        end

        loss(w, state, words, ys; lss=nll, views=views, as=acts, dropout=false, args=args)

        lss += nll[1] * nll[2]
        cnt += nll[2]
    end
    return lss / cnt
end

function test(models, data, maps; args=nothing)
    scss = 0.0

    for (instruction, words) in data
        words = map(v->convert(KnetArray{Float32},v), words)
        states = map(weights->initstate(KnetArray{Float32}, args["hidden"], 1, length(words)), models)

        for ind=1:length(models)
            w = models[ind]
            state = states[ind]
            encode(w["enc_w_f"], w["enc_b_f"], w["enc_w_b"], w["enc_b_b"], w["emb_word"],
                state, words; dropout=false)

            state[5] = hcat(state[1][end], state[3][end])
            state[6] = hcat(state[2][end], state[4][end])
        end

        current = instruction.path[1]
        nactions = 0
        stop = false

        actions = Any[]
        araw = args["preva"] ? reshape(Float32[0.0, 0.0, 0.0, 1.0], 1, 4) : nothing
        while !stop
            preva = araw != nothing ? convert(KnetArray{Float32}, araw)  : araw

            view = !args["percp"] ? nothing : args["encoding"] == "grid" ? 
                state_agent_centric(maps[instruction.map], current) : state_agent_centric_multihot(maps[instruction.map], current)
            view = args["percp"] ? convert(KnetArray{Float32}, view) : nothing

            cum_ps = zeros(Float32, 1, 4)
            for ind=1:length(models)
                w = models[ind]
                state = states[ind]

                att,_ = args["att"] || args["attinwatt"] != 0 ? attention(state, w["attention_w"], w["attention_v"]) : (nothing, nothing)

                if !args["percp"]
                    x = spatial(w["emb_world"], preva)
                elseif args["encoding"] == "grid" && args["worldatt"] != 0
                    wattinp = args["attinwatt"] == 0 ? state[5] : hcat(state[5], att)
                    worldatt = worldattention(wattinp, w["wa1"], w["wa2"])
                    x =  spatial(w["filters_w"], w["filters_b"], w["emb_world"], worldatt, view) # world att
                elseif args["encoding"] == "grid" 
                    x = spatial(w["filters_w"], w["filters_b"], w["emb_world"], view)
                else
                    x = spatial(w["emb_world"], view)
                end

                soft_inp = args["inpout"] ? w["soft_inp"] : nothing
                soft_att = args["attout"] ? w["soft_att"] : nothing
                soft_preva = args["prevaout"] ? w["soft_preva"] : nothing
                preva = !args["preva"] ? nothing : preva
                prevainp = args["preva"] && args["percp"]

                att = args["att"] ? att : nothing
                ypred = decode(w["dec_w"], w["dec_b"], w["soft_h"], w["soft_b"], state, x; soft_inp=soft_inp, soft_att=soft_att, 
                soft_preva=soft_preva, preva=preva, att=att, dropout=false, prevainp=prevainp)

                cum_ps += probs(Array(ypred))
            end

            cum_ps = cum_ps ./ length(models)
            debug("Probs: $(cum_ps)")
            action = 0
            if args["greedy"]
                action = indmax(cum_ps)
            else
                action = sample(cum_ps)
            end
            
            if args["preva"]
                araw[:] = 0.0
                araw[1, action] = 1.0
            end

            push!(actions, action)
            prev = current
            current = getlocation(maps[instruction.map], current, action)
            nactions += 1

            nowall = false
            if action == 1
                nowall = !haskey(maps[instruction.map].edges[(prev[1], prev[2])], (current[1], current[2]))
            end

            stop = nactions > args["limactions"] || action == 4 || nowall

        end

        debug("$(instruction.text)")
        debug("Path: $(instruction.path)")
        debug("Filename: $(instruction.fname)")

        debug("Actions: $(reshape(collect(actions), 1, length(actions)))")
        debug("Current: $(current)")

        if current == instruction.path[end]
            scss += 1
            debug("SUCCESS\n")
        else
            debug("FAILURE\n")
        end
    end

    return scss / length(data)
end

function test_paragraph(models, groups, maps; args=nothing)
    scss = 0.0
    for data in groups
        debug("\nNew paragraph")
        current = data[1][1].path[1]
        
        for i=1:length(data)
            instruction, words = data[i]
            words = map(v->convert(KnetArray{Float32},v), words)
            states = map(weights->initstate(KnetArray{Float32}, args["hidden"], 1, length(words)), models)

            for ind=1:length(models)
                w = models[ind]
                state = states[ind]
                encode(w["enc_w_f"], w["enc_b_f"], w["enc_w_b"], w["enc_b_b"], w["emb_word"],
                state, words; dropout=false)

                state[5] = hcat(state[1][end], state[3][end])
                state[6] = hcat(state[2][end], state[4][end])
            end

            nactions = 0
            stop = false
            action = 0
            actions = Any[]
            araw = args["preva"] ? reshape(Float32[0.0, 0.0, 0.0, 1.0], 1, 4) : nothing
            while !stop
                preva = araw != nothing ? convert(KnetArray{Float32}, araw)  : araw

                view = !args["percp"] ? nothing : args["encoding"] == "grid" ? 
                state_agent_centric(maps[instruction.map], current) : state_agent_centric_multihot(maps[instruction.map], current)
                view = args["percp"] ? convert(KnetArray{Float32}, view) : nothing

                cum_ps = zeros(Float32, 1, 4)
                for ind=1:length(models)
                    w = models[ind]
                    state = states[ind]

                    att,_ = args["att"] || args["attinwatt"] != 0 ? attention(state, w["attention_w"], w["attention_v"]) : (nothing, nothing)

                    if !args["percp"]
                        x = spatial(w["emb_world"], preva)
                    elseif args["encoding"] == "grid" && args["worldatt"] != 0
                        wattinp = args["attinwatt"] == 0 ? state[5] : hcat(state[5], att)
                        worldatt = worldattention(wattinp, w["wa1"], w["wa2"])
                        x =  spatial(w["filters_w"], w["filters_b"], w["emb_world"], worldatt, view) # world att
                    elseif args["encoding"] == "grid" 
                        x = spatial(w["filters_w"], w["filters_b"], w["emb_world"], view)
                    else
                        x = spatial(w["emb_world"], view)
                    end

                    soft_inp = args["inpout"] ? w["soft_inp"] : nothing
                    soft_att = args["attout"] ? w["soft_att"] : nothing
                    soft_preva = args["prevaout"] ? w["soft_preva"] : nothing
                    preva = !args["preva"] ? nothing : preva
                    prevainp = args["preva"] && args["percp"]

                    att = args["att"] ? att : nothing
                    ypred = decode(w["dec_w"], w["dec_b"], w["soft_h"], w["soft_b"], state, x; soft_inp=soft_inp, soft_att=soft_att, 
                    soft_preva=soft_preva, preva=preva, att=att, dropout=false, prevainp=prevainp)

                    cum_ps += probs(Array(ypred))
                end

                cum_ps = cum_ps ./ length(models)
                debug("Probs: $(cum_ps)")
                if args["greedy"]
                    action = indmax(cum_ps)
                else
                    action = sample(cum_ps)
                end

                if args["preva"]
                    araw[:] = 0.0
                    araw[1, action] = 1.0
                end

                push!(actions, action)
                prev = current
                current = getlocation(maps[instruction.map], current, action)
                nactions += 1

                nowall = false
                if action == 1
                    nowall = !haskey(maps[instruction.map].edges[(prev[1], prev[2])], (current[1], current[2]))
                end

                stop = nactions > args["limactions"] || action == 4 || nowall
            end

            debug("$(instruction.text)")
            debug("Path: $(instruction.path)")
            debug("Filename: $(instruction.fname)")

            debug("Actions: $(reshape(collect(actions), 1, length(actions)))")
            debug("Current: $(current)")

            if action != 4
                debug("FAILURE")
                break
            end

            if i == length(data)
                if current[1] == instruction.path[end][1] && current[2] == instruction.path[end][2]
                    scss += 1
                    debug("SUCCESS\n")
                else
                    debug("FAILURE\n")
                end
            end
        end
    end
    return scss / length(groups)
end

function initparams(ws; args=nothing)
    prms = Dict()

    for k in keys(ws)
        if startswith(k, "filter")
            prms[k] = map(w->Adam(lr=args["lr"]), ws[k])
            #prms[k] = map(w->Sgd(;lr=args["lr"]), ws[k])
        else
            prms[k] = Adam(lr=args["lr"]) 
            #prms[k] = Sgd(;lr=args["lr"]) 
        end
    end

    return prms
end

# state[2k-1,2k]: hidden and cell for the k'th lstm layer
function initstate(atype, hidden, batchsize, length)
    state = Array{Any}(6)
    #forward
    state[1] = Array{Any}(length+1)
    for i=1:(length+1); state[1][i] = convert(atype, zeros(batchsize, hidden)); end

    state[2] = Array{Any}(length+1)
    for i=1:(length+1); state[2][i] = convert(atype, zeros(batchsize, hidden)); end

    #backward
    state[3] = Array{Any}(length+1)
    for i=1:(length+1); state[3][i] = convert(atype, zeros(batchsize, hidden)); end

    state[4] = Array{Any}(length+1)
    for i=1:(length+1); state[4][i] = convert(atype, zeros(batchsize, hidden)); end

    state[5] = convert(atype, zeros(batchsize, hidden*2))
    state[6] = convert(atype, zeros(batchsize, hidden*2))

    return state
end

function test_beam(models, data, maps; args=nothing)
    beamsize = args["beamsize"]

    scss = 0.0

    for (instruction, words) in data
        words = map(v->convert(KnetArray{Float32},v), words)
        states = map(weights->initstate(KnetArray{Float32}, args["hidden"], 1, length(words)), models)

        for i=1:length(models)
            w = models[i]
            state = states[i]
            encode(w["enc_w_f"], w["enc_b_f"], w["enc_w_b"], w["enc_b_b"], w["emb_word"], state, words; dropout=false)

            state[5] = hcat(state[1][end], state[3][end])
            state[6] = hcat(state[2][end], state[4][end])
        end
        
        cstate5 = Any[]
        cstate6 = Any[]
        
        for m=1:length(models)
            push!(cstate5, copy(states[m][5]))
            push!(cstate6, copy(states[m][6]))
        end

        current = instruction.path[1]
        cands = Any[(1.0, cstate5, cstate6, current, 0, false, Any[4])]

        nactions = 0
        stop = false
        stopsearch = false
        araw = args["preva"] ? reshape(Float32[0.0, 0.0, 0.0, 1.0], 1, 4) : nothing
        while !stopsearch
            newcands = Any[]
            newcand = false
            for cand in cands
                current = cand[4]
                for m=1:length(models)
                    states[m][5] = copy(cand[2][m])
                    states[m][6] = copy(cand[3][m])
                end
                depth = cand[5]
                prevActions = cand[end]
                lastAction = prevActions[end]
                stopped = cand[6]

                if araw != nothing
                    araw[:] = 0.0
                    araw[1, lastAction] = 1.0
                end

                if stopped
                    push!(newcands, cand)
                    continue
                end

                view = !args["percp"] ? nothing : args["encoding"] == "grid" ? 
                    state_agent_centric(maps[instruction.map], current) : state_agent_centric_multihot(maps[instruction.map], current)
                view = args["percp"] ? convert(KnetArray{Float32}, view) : nothing
                preva = araw != nothing ? convert(KnetArray{Float32}, araw)  : araw

                cum_ps = zeros(Float32, 1, 4)
                #score = KnetArray(zeros(Float32, 1, 4))

                for ind=1:length(models)
                    w = models[ind]
                    state = states[ind]

                    att,_ = args["att"] || args["attinwatt"] != 0 ? attention(state, w["attention_w"], w["attention_v"]) : (nothing, nothing)

                    if !args["percp"]
                        x = spatial(w["emb_world"], preva)
                    elseif args["encoding"] == "grid" && args["worldatt"] != 0
                        wattinp = args["attinwatt"] == 0 ? state[5] : hcat(state[5], att)
                        worldatt = worldattention(wattinp, w["wa1"], w["wa2"])
                        x =  spatial(w["filters_w"], w["filters_b"], w["emb_world"], worldatt, view) # world att
                    elseif args["encoding"] == "grid" 
                        x = spatial(w["filters_w"], w["filters_b"], w["emb_world"], view)
                    else
                        x = spatial(w["emb_world"], view)
                    end

                    soft_inp = args["inpout"] ? w["soft_inp"] : nothing
                    soft_att = args["attout"] ? w["soft_att"] : nothing
                    soft_preva = args["prevaout"] ? w["soft_preva"] : nothing
                    preva = !args["preva"] ? nothing : preva
                    prevainp = args["preva"] && args["percp"]

                    att = args["att"] ? att : nothing
                    ypred = decode(w["dec_w"], w["dec_b"], w["soft_h"], w["soft_b"], state, x; soft_inp=soft_inp, soft_att=soft_att, 
                    soft_preva=soft_preva, preva=preva, att=att, dropout=false, prevainp=prevainp)
                    cum_ps += probs(Array(ypred))
                    #score += ypred
                end

                cum_ps = cum_ps ./ length(models)
                #score = score ./ length(models)
                debug("Probs: $(cum_ps)")
                #debug("Score: $(Array(score))")

                for i=1:4
                    mystop = stopped
                    nactions = depth + 1
                    actcopy = copy(prevActions)
                    if nactions > args["limactions"] && i < 4
                        mystop = true
                    end
                    push!(actcopy, i)
                    cur = identity(current)
                    if i < 4
                        cur = getlocation(maps[instruction.map], cur, i)
                    end

                    if (i == 1 && !haskey(maps[instruction.map].edges[(current[1], current[2])], (cur[1], cur[2]))) || i==4
                        mystop = true
                    end

                    cstate5 = Any[]
                    cstate6 = Any[]
                    for m=1:length(models)
                        push!(cstate5, copy(states[m][5]))
                        push!(cstate6, copy(states[m][6]))
                    end

                    push!(newcands, (cand[1] * cum_ps[1, i], cstate5, cstate6, cur, nactions, mystop, actcopy))
                    #push!(newcands, (cand[1] + score[1, i], cstate5, cstate6, cur, nactions, mystop, actcopy))
                    newcand = true
                end
            end
            stopsearch = !newcand
            if newcand
                sort!(newcands; by=x->x[1], rev=true)
                l = length(newcands) < beamsize ? length(newcands) : beamsize
                cands = newcands[1:l]
            end
        end
        current = cands[1][4]
        actions = cands[1][end]
        debug("$(instruction.text)")
        debug("Path: $(instruction.path)")
        debug("Filename: $(instruction.fname)")

        debug("Actions: $(reshape(collect(actions), 1, length(actions)))")
        debug("Current: $(current)")
        debug("Prob: $(cands[1][1])")
        #debug("Score: $(cands[1][1])")

        if current == instruction.path[end]
            scss += 1
            debug("SUCCESS\n")
        else
            debug("FAILURE\n")
        end
    end

    return scss / length(data)
end

function test_paragraph_beam(models, groups, maps; args=nothing)
    beamsize = args["beamsize"]

    scss = 0.0

    for data in groups
        debug("\nNew paragraph")
        current = data[1][1].path[1]
        cands = Any[]
        for indx=1:length(data)
            instruction, words = data[indx]
            words = map(v->convert(KnetArray{Float32},v), words)
            states = map(weights->initstate(KnetArray{Float32}, args["hidden"], 1, length(words)), models)

            for ind=1:length(models)
                w = models[ind]
                state = states[ind]
                encode(w["enc_w_f"], w["enc_b_f"], w["enc_w_b"], w["enc_b_b"], w["emb_word"], state, words; dropout=false)

                state[5] = hcat(state[1][end], state[3][end])
                state[6] = hcat(state[2][end], state[4][end])
            end
            cstate5 = Any[]
            cstate6 = Any[]
            for m=1:length(models)
                push!(cstate5, copy(states[m][5]))
                push!(cstate6, copy(states[m][6]))
            end

            if length(cands) == 0
                cands = Any[(1.0, cstate5, cstate6, current, 0, false, true, Any[4])]
            else
                cands = map(x->(x[1], cstate5, cstate6, x[4], 0, false, x[7], x[end]), cands)
            end

            nactions = 0
            stop = false
            stopsearch = false

            araw = args["preva"] ? reshape(Float32[0.0, 0.0, 0.0, 1.0], 1, 4) : nothing
            while !stopsearch
                newcands = Any[]
                newcand = false
                for cand in cands
                    current = cand[4]
                    for m=1:length(models)
                        states[m][5] = copy(cand[2][m])
                        states[m][6] = copy(cand[3][m])
                    end
                    depth = cand[5]
                    prevActions = cand[end]
                    lastAction = prevActions[end]
                    stopped = cand[6]
                    if araw != nothing
                        araw[:] = 0.0
                        araw[1, lastAction] = 1.0
                    end

                    if stopped || !cand[7]
                        c = (cand[1], cand[2], cand[3], cand[4], cand[5], true, cand[7], cand[end])
                        push!(newcands, c)
                        continue
                    end

                    preva = araw != nothing ? convert(KnetArray{Float32}, araw)  : araw

                    view = !args["percp"] ? nothing : args["encoding"] == "grid" ? 
                    state_agent_centric(maps[instruction.map], current) : state_agent_centric_multihot(maps[instruction.map], current)
                    view = args["percp"] ? convert(KnetArray{Float32}, view) : nothing

                    cum_ps = zeros(Float32, 1, 4)

                    for ind=1:length(models)
                        w = models[ind]
                        state = states[ind]

                        att,_ = args["att"] || args["attinwatt"] != 0 ? attention(state, w["attention_w"], w["attention_v"]) : (nothing, nothing)

                        if !args["percp"]
                            x = spatial(w["emb_world"], preva)
                        elseif args["encoding"] == "grid" && args["worldatt"] != 0
                            wattinp = args["attinwatt"] == 0 ? state[5] : hcat(state[5], att)
                            worldatt = worldattention(wattinp, w["wa1"], w["wa2"])
                            x =  spatial(w["filters_w"], w["filters_b"], w["emb_world"], worldatt, view) # world att
                        elseif args["encoding"] == "grid" 
                            x = spatial(w["filters_w"], w["filters_b"], w["emb_world"], view)
                        else
                            x = spatial(w["emb_world"], view)
                        end

                        soft_inp = args["inpout"] ? w["soft_inp"] : nothing
                        soft_att = args["attout"] ? w["soft_att"] : nothing
                        soft_preva = args["prevaout"] ? w["soft_preva"] : nothing
                        preva = !args["preva"] ? nothing : preva
                        prevainp = args["preva"] && args["percp"]

                        att = args["att"] ? att : nothing
                        ypred = decode(w["dec_w"], w["dec_b"], w["soft_h"], w["soft_b"], state, x; soft_inp=soft_inp, soft_att=soft_att, 
                        soft_preva=soft_preva, preva=preva, att=att, dropout=false, prevainp=prevainp)
                        cum_ps += probs(Array(ypred))
                    end

                    cum_ps = cum_ps ./ length(models)
                    debug("Probs: $(cum_ps)")
                    for i=1:4
                        mystop = stopped
                        legitimate = true
                        nactions = depth + 1
                        actcopy = copy(prevActions)
                        if nactions > args["limactions"] && i < 4
                            mystop = true
                            legitimate = false
                        end
                        push!(actcopy, i)
                        cur = identity(current)
                        if i < 4
                            cur = getlocation(maps[instruction.map], cur, i)
                        end

                        if (i == 1 && !haskey(maps[instruction.map].edges[(current[1], current[2])], (cur[1], cur[2]))) || i==4
                            mystop = true
                            if i == 1
                                legitimate = false
                            end
                        end

                        cstate5 = Any[]
                        cstate6 = Any[]
                        for m=1:length(models)
                            push!(cstate5, copy(states[m][5]))
                            push!(cstate6, copy(states[m][6]))
                        end

                        push!(newcands, (cand[1] * cum_ps[1, i], cstate5, cstate6, cur, nactions, mystop, legitimate, actcopy))
                        newcand = true
                    end
                end
                stopsearch = !newcand
                if !stopsearch
                    sort!(newcands; by=x->x[1], rev=true)
                    l = length(newcands) < beamsize ? length(newcands) : beamsize
                    cands = newcands[1:l]
                end
            end

            debug("$(instruction.text)")
            debug("Path: $(instruction.path)")
            debug("Filename: $(instruction.fname)")

            if indx == length(data)
                current = cands[1][4]
                actions = cands[1][end]
                debug("Actions: $(reshape(collect(actions), 1, length(actions)))")
                debug("Current: $(current)")
                debug("Prob: $(cands[1][1])")

                if actions[end] != 4
                    debug("FAILURE")
                    break
                end

                if current[1] == instruction.path[end][1] && current[2] == instruction.path[end][2]
                    scss += 1
                    debug("SUCCESS\n")
                else
                    debug("FAILURE\n")
                end
            end
        end
    end

    return scss / length(groups)
end
