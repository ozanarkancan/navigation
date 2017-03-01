# TODO: dropout, training limit, external embeddings (under data)

using Knet, ArgParse, JLD

# Using Int[] for input/output sequences.
# Will rethink when we do minibatching.


"""
    lstm(weight,bias,hidden,cell,input) => (newhidden,newcell)
    
All inputs are row major.
"""
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


"""
    dropout(x,p)

Return `x` if `p==0` or new array with `p` fraction of elements in `x` replaced with 0's if `p>0`.
"""
function dropout(x,p)
    if p > 0
        x .* (rand!(similar(x)) .> p) ./ (1-p)
    else
        x
    end
end


"""
    encode(param, state, input) => newstate

* `input`: an integer representing an input token
* `state[2k-1,2k]`: hidden and cell for the k'th lstm layer
* `param[2k-1,2k]`: weight and bias for k'th lstm layer
* `param[1+length(state)]`: input embeddings (array of row vectors)
"""
function encode(param, state, input; pdrop=0)
    input = param[1+length(state)][input]
    newstate = similar(state)
    for i = 1:2:length(state)
        input = dropout(input, pdrop)
        (newstate[i],newstate[i+1]) = lstm(param[i],param[i+1],state[i],state[i+1],input)
        input = newstate[i]
    end
    return newstate
end


"""
    decode(param, state, input) => prediction,newstate

* `input`: an integer representing an input token
* `state[2k-1,2k]`: hidden and cell for the k'th lstm layer
* `param[2k-1,2k]`: weight and bias for k'th lstm layer
* `param[end-2]`: input embeddings (Vector of row vectors)
* `param[end-1,end]`: weight and bias for final prediction
"""
function decode(param, state, input; pdrop=0)
    newstate = encode(param, state, input; pdrop=pdrop)
    output = dropout(newstate[end-1], pdrop)
    prediction = output * param[end-1] .+ param[end]
    return (prediction, newstate)
end


"""
    s2s(params,state,inputs,outputs,EOS)

Return loss for the [sequence-to-sequence model](https://arxiv.org/abs/1409.3215).

* `params`: a pair with encoder and decoder parameters.
* `state`: initial hidden and cell states.
* `inputs::Vector{Int}`: Input sequence for the encoder. No start/end tokens.
* `outputs::Vector{Int}`: Output sequence for the decoder. No start/end tokens.
"""
function s2s(params,state,inputs,outputs; EOS=1, pdrop=0)
    encoder,decoder = params
    for input in reverse(inputs)
        state = encode(encoder, state, input; pdrop=pdrop)
    end
    sumloss = 0
    prev = EOS
    for output in outputs
        prediction,state = decode(decoder, state, prev; pdrop=pdrop)
        sumloss -= logp(prediction,2)[output]
        prev = output

    end
    prediction,state = decode(decoder, state, prev; pdrop=pdrop)
    sumloss -= logp(prediction,2)[EOS]
    return sumloss
end

s2sgrad = grad(s2s)

"""
    avgloss(params,data)

Return average per-token loss over all (x,y) sequence pairs in data.    
"""
function avgloss(params,data)
    sumloss = cntloss = 0
    state = initstate(params[1])
    for (inputs,outputs) in data
        sumloss += s2s(params,state,inputs,outputs)
        cntloss += length(outputs)+1 # +1 for EOS
    end
    return sumloss / cntloss
end

Base.indmax(x::KnetArray)=indmax(Array(x))


"""
    predict(params,state,inputs)

Return an output sequence for a given a model, initial state, and an
input sequence.  The sequences are Array{Int} and do not contain EOS
tokens.

"""
function predict(params,state,inputs; EOS=1, maxlen=10)
    encoder,decoder = params
    for input in reverse(inputs)
        state = encode(encoder, state, input)
    end
    action = EOS; actions = Int[]
    while true
        pred,state = decode(decoder, state, action)
        action = indmax(pred)
        if action == EOS || length(actions) >= maxlen; break; end
        push!(actions, action)
    end
    return actions
end


"""
    accuracy(params,data)

Return percentage of correctly predicted action sequences.  Only exact
match output sequences considered correct.

"""    
function accuracy(params,data)
    correct = total = 0f0
    state = initstate(params[1])
    for (inputs,outputs) in data
        p = predict(params, state, inputs)
        correct += (p == outputs)
        total += 1
    end
    return correct / total
end


function readdata2(file, widx1, widx2)
    data = []
    for line in eachline(file)
        line = chomp(line)
        input,output = split(line, '\t')
        isempty(input) && isempty(output) && continue
        x = Int[]
        for w in split(input)
            push!(x, get!(widx1, w, 1+length(widx1)))
        end
        y = Int[]
        for w in split(output)
            push!(y, get!(widx2, w, 1+length(widx2)))
        end
        push!(data, (x,y))
    end
    return data
end

# `param[2k-1,2k]`: weight and bias for k'th lstm layer
# `param[end-2]`: input embedding matrix
# `param[end-1,end]`: weight and bias for final prediction
function initweights(; hidden=nothing, embed1=nothing, embed2=nothing,
                       atype=nothing,  vocab1=nothing, vocab2=nothing, o...)
    function mkemb(emb,voc)
        a = Array(atype,length(voc))
        if isa(emb,Dict)
            length(emb) == length(voc) || error("Embedding and vocab sizes mismatch")
            for k in keys(emb)
                haskey(voc,k) || error("Embedding and vocab keys mismatch")
                a[voc[k]] = atype(reshape(emb[k],1,length(emb[k])))
            end
        elseif isa(emb,Int)
            for i=1:length(a)
                a[i] = atype(xavier(1,emb))
            end
        else
            error("Unrecognized embedding $(summary(emb))")
        end
        return a
    end
    nlayer = length(hidden)
    encoder = Array(Any, 2*nlayer+1)
    decoder = Array(Any, 2*nlayer+3)
    encoder[2*nlayer+1] = mkemb(embed1,vocab1)
    decoder[2*nlayer+1] = mkemb(embed2,vocab2)
    input1 = length(encoder[2*nlayer+1][1])
    input2 = length(decoder[2*nlayer+1][1])
    for k = 1:nlayer
        encoder[2k-1] = atype(xavier(input1 + hidden[k], 4*hidden[k]))
        decoder[2k-1] = atype(xavier(input2 + hidden[k], 4*hidden[k]))
        encoder[2k] = atype(zeros(1, 4*hidden[k]))
        decoder[2k] = atype(zeros(1, 4*hidden[k])) # TODO: do we really need biases?
        encoder[2k][1:hidden[k]] = 1 # forget gate bias init=1
        decoder[2k][1:hidden[k]] = 1 # forget gate bias init=1
        input1 = input2 = hidden[k]
    end
    decoder[2*nlayer+2] = atype(xavier(hidden[end],length(vocab2)))
    decoder[2*nlayer+3] = atype(zeros(1, length(vocab2)))
    return (encoder, decoder)
end

function initstate(encoder)
    state = Array(Any, length(encoder)-1)
    batch = 1 # size(input,1)
    for i=1:2:length(state)
        hsize = div(size(encoder[i],2),4)
        state[i]   = fill!(similar(encoder[i], (batch,hsize)), 0)
        state[i+1] = fill!(similar(encoder[i], (batch,hsize)), 0)
    end
    return state
end

# This should work for any combination of tuple/array
oparams{T<:Number}(::KnetArray{T})=Adam()
oparams{T<:Number}(::Array{T})=Adam()
oparams(x)=map(oparams, x)

# Print hierarchical structure for debugging
function pstruct(x,r=0)
    print(repeat(" ",r))
    println(summary(x))
    if (isa(x,Array)||isa(x,Tuple)) && !isbits(eltype(x)) && length(x) <= 10
        for xi in x
            pstruct(xi,r+4)
        end
    end
end


# hyperband stuff
# include(Knet.dir("examples/hyperband.jl"))
# config0 = main("--fast --epochs 0 --train trn tst")

function getconfig2()
    c = Dict()
    c[:hidden] = [ rand(50:200) ] # multi-layer? dropout?, gclip?
    c[:embed1] = rand(100:600)
    c[:embed2] = rand(10:100)
    return c
end

function getloss2(c,n)
    global config0
    if !isdefined(:config0); config0 = main("--fast --epochs 0 --train trn tst"); end
    o = copy(config0)
    for s in (:hidden,:embed1,:embed2); o[s] = c[s]; end
    o[:epochs] = n
    o[:model] = initweights(; o...)
    o[:opt] = oparams(o[:model])
    s0 = initstate(o[:model][1])
    for epoch = 1:o[:epochs]
        for (x,y) in o[:datas][1]
            o[:grad] = s2sgrad(o[:model],s0,x,y)
            Knet.update!(o[:model],o[:grad],o[:opt])
        end
    end
    loss = avgloss(o[:model],o[:datas][2])
    println((n, loss, [(k,v) for (k,v) in c]...))
    return loss
end

# Navigation specific part

include("util.jl")
include("io.jl")
function savemaps()
    instrs = getallinstructions()
    fnames = ["grid.txt", "jelly.txt", "l.txt"]
    actions = ["move","right","left","stop"]
    for i=1:3
        io = open(fnames[i],"w")
        for instr in instrs[i]
            print(io, join(instr.text,' '))
            print(io, '\t')
            a = getactions(instr.path)
            pop!(a) # s2s adds stop
            print(io, join(actions[a],' '))
            print(io, '\n')
        end
        close(io)
    end
end


function main(args=ARGS)
    s = ArgParseSettings()
    s.description="s2s.jl [sequence-to-sequence model](https://arxiv.org/abs/1409.3215). (c) Deniz Yuret, 2017."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="Random number seed.")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="Array and element type.")
        ("--train"; nargs='+'; help="If provided, use first file for training, second for dev, others for eval.")
        ("--hidden"; nargs='+'; arg_type=Int; default=[64]; help="Sizes of one or more LSTM layers.")
        ("--embed1"; arg_type=Int; default=512; help="Size of the input embedding vector.")
        ("--embed2"; arg_type=Int; default=32;  help="Size of the output embedding vector.")
        ("--epochs"; arg_type=Int; default=10; help="Number of epochs for training.")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--loadfile"; help="Initialize model, vocab, and/or embeddings from file")
        ("--dropout"; arg_type=Float32; default=0.5f0; help="Dropout probability.")
        # TODO:
        # ("--test"; help="Apply model to input sequences in test file.")
        # ("--savefile"; help="Save final model to file")
        # ("--bestfile"; help="Save best model to file")
        # ("--batchsize"; arg_type=Int; default=10; help="Size of minibatches.")
        # ("--gclip"; arg_type=Float64; default=0.0; help="Gradient clip.")
    end
    println(s.description)
    isa(args, AbstractString) && (args=split(args))
    global o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && setseed(o[:seed])
    o[:atype] = eval(parse(o[:atype]))
    o[:vocab1] = Dict(); o[:vocab2] = Dict(" <S> "=>1) # TODO: handle unk, eos, restricted vocab size
    if o[:loadfile] != nothing
        for (k,v) in load(o[:loadfile])
            o[Symbol(k)] = v
        end
    end
    o[:datas] = [readdata2(file,o[:vocab1],o[:vocab2]) for file in o[:train]]
    if !haskey(o,:model); o[:model] = initweights(; o...); end
    o[:opt] = oparams(o[:model])
    @printf("vocab1=%d vocab2=%d embed1=%d embed2=%d\n",
            length(o[:vocab1]), length(o[:vocab2]),
            length(o[:model][1][end][1]), length(o[:model][2][end-2][1]))
    # o[:models] = [ deepcopy(o[:model]) ]
    report(t,p,d)=println((t,[ (avgloss(p,di),accuracy(p,di)) for di in d ]...))
    !o[:fast] && report(0,o[:model],o[:datas])
    istate = initstate(o[:model][1])
    for epoch = 1:o[:epochs]
        @time for (x,y) in o[:datas][1]
            o[:grad] = s2sgrad(o[:model],istate,x,y; pdrop=o[:dropout])
            Knet.update!(o[:model],o[:grad],o[:opt])
        end
        # push!(o[:models], deepcopy(o[:model]))
        !o[:fast] && report(epoch,o[:model],o[:datas])
    end
    return o
end


# Experiments:

# Baseline:
# main("--train trn tst --seed 1")
# trn: jelly+l, tst: grid
# opts=(:atype,"KnetArray{Float32}")(:train,Any["trn","tst"])(:winit,0.1)(:vocab2,nothing)(:embed2,32)(:hidden,[64])(:epochs,5)(:seed,1)(:vocab1,nothing)(:embed1,512)(:fast,false)
# (2,0.5125578f0,0.6423336f0)

# Xavier:
# (3,0.49982417f0,0.64756024f0)
# Equal performance, one less parameter, adapted.

# Reverse input:
# (3,0.48766702f0,0.62641555f0), accuracy: 0.6167048054919908
# Significant, adapted.

# Initializing with one-hot input embeddings:
#    onehot(n,i)=(x=zeros(1,n);x[i]=1;x)
#    encoder[2*nlayer+1] = [ atype(onehot(length(vocab1),i)) for i=1:length(vocab1) ]
# (3,0.49662954f0,0.6173695f0)
# Using xavier with embed1=vocabsize instead:
# (3,0.47787347f0,0.62890613f0)
# Initializing onehot embeddings for both encoder and decoder inputs:
#    decoder[2*nlayer+1] = [ atype(onehot(length(vocab2),i)) for i=1:length(vocab2) ]
# (3,0.5062695f0,0.6270728f0)
# Always using onehot embeddings:
# (2,0.5919869f0,0.65982556f0)
#    i = input; input = similar(param[1+length(state)][i])
#    fill!(input,0); input[i]=1
# This does not really generalize, not worth it.

# Load pretrained word embeddings:
# o = main("--train trn tst --load embed1.jld --seed 1");
# best epoch=3 avgloss=0.60410120f0 accuracy=0.62128
# compared to random embeddings:
# best epoch=3 avgloss=0.62535596f0 accuracy=0.61670
# adapted as an option.

# Initialize forget gate bias = 1:
# o = main("--train trn tst --seed 1");
# vocab1=524 vocab2=4 embed1=512 embed2=32
# best epoch=2 avgloss=0.6265209f0
# Not significant difference but right thing to do, adapted.

# Pretrained embed with latest version:
# o = main("--train trn tst --seed 1 --load embed1.jld")
# vocab1=524 vocab2=4 embed1=300 embed2=32
# best avgloss=0.6006409f0 at epoch=3
# best accuracy=0.61327231 at epoch=2

# Test loss and accuracy do not reach minimum together:
# 
# o = main("--train trn tst --seed 1") # default options
# (epoch,(trnloss,trnacc),(tstloss,tstacc))
# (5,(0.39382565f0,0.7431232f0),(0.6655941f0,0.6235698f0))  # best accuracy
# (2,(0.5277382f0,0.66271687f0),(0.6265209f0,0.60297483f0)) # best loss
# 
# o = main("--train trn tst --seed 1 --dropout 0.5") # dropout with best test accuracy (tried 0.1:0.1:0.9)
# (6,(0.44146112f0,0.71603894f0),(0.6252134f0,0.6361556f0)) # best accuracy at dropout=0.5 epoch=6
# (4,(0.48044023f0,0.69318664f0),(0.60365015f0,0.61899316f0)) # best loss at dropout=0.5 epoch=4
#
# o = main("--train trn tst --seed 1 --load embed1.jld") # pretrained embeddings
# (7,(0.32677525f0,0.78374946f0),(0.74263537f0,0.61899316f0)) # best accuracy
# (3,(0.4598019f0,0.70672876f0),(0.6006409f0,0.6075515f0)) # best loss
#
# o = main("--train trn tst --seed 1 --load embed1.jld --dropout ?") # pretrained embeddings with dropout
# (6,(0.44951436f0,0.70799834f0),(0.5994176f0,0.63386726f0)) # best accurary at dropout=0.6 epoch=6
# (3,(0.48624966f0,0.68303007f0),(0.59056264f0,0.617849f0))  # best loss at dropout=0.3 epoch=3
