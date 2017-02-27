using Knet, ArgParse, JLD

# Using Int[] for input/output sequences.
# Will rethink when we do minibatching.

# Some special tokens
const EOS=1
const EOSSTR=" S "
# const UNK=2
# const UNKSTR=" U "
# spaces ensure they won't match regular tokens which are stripped

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
    encode(param, state, input) => newstate

* `input`: an integer representing an input token
* `state[2k-1,2k]`: hidden and cell for the k'th lstm layer
* `param[2k-1,2k]`: weight and bias for k'th lstm layer
* `param[1+length(state)]`: input embeddings (array of row vectors)
"""
function encode(param, state, input)
    input = param[1+length(state)][input]
    newstate = similar(state)
    for i = 1:2:length(state)
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
* `param[end-2]`: input embeddings (array of row vectors)
* `param[end-1,end]`: weight and bias for final prediction
"""
function decode(param, state, input)
    newstate = encode(param, state, input)
    prediction = newstate[end-1] * param[end-1] .+ param[end]
    return (prediction, newstate)
end


"""
    s2s(params,state,inputs,outputs)

Return loss for the [sequence-to-sequence model](https://arxiv.org/abs/1409.3215).

* `params`: a pair with encoder and decoder parameters.
* `state`: initial hidden and cell states.
* `inputs`: input sequence for the encoder. Int array, no start/end tokens.
* `outputs`: output sequence for the decoder. Int array, no start/end tokens.
"""
function s2s(params,state,inputs,outputs)
    encoder,decoder = params
    for input in reverse(inputs)
        state = encode(encoder, state, input)
    end
    sumloss = 0
    prev = EOS
    for output in outputs
        prediction,state = decode(decoder, state, prev)
        sumloss -= logp(prediction,2)[output]
        prev = output

    end
    prediction,state = decode(decoder, state, prev)
    sumloss -= logp(prediction,2)[EOS]
    return sumloss
end

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
function predict(params,state,inputs)
    encoder,decoder = params
    for input in reverse(inputs)
        state = encode(encoder, state, input)
    end
    action = EOS; actions = Int[]
    while true
        pred,state = decode(decoder, state, action)
        action = indmax(pred)
        if action == EOS || length(actions) > 10; break; end
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
    correct = total = 0
    state = initstate(params[1])
    for (inputs,outputs) in data
        p = predict(params, state, inputs)
        correct += (p == outputs)
        total += 1
    end
    return correct / total
end


function readdata2(file; vocab1=nothing, vocab2=nothing, o...)
    data = []
    for line in eachline(file)
        line = chomp(line)
        input,output = split(line, '\t')
        isempty(input) && isempty(output) && continue
        x = Int[]
        for w in split(input)
            push!(x, get!(vocab1, w, 1+length(vocab1)))
        end
        y = Int[]
        for w in split(output)
            push!(y, get!(vocab2, w, 1+length(vocab2)))
        end
        push!(data, (x,y))
    end
    return data
end

# `param[2k-1,2k]`: weight and bias for k'th lstm layer
# `param[end-2]`: input embedding matrix
# `param[end-1,end]`: weight and bias for final prediction
function initweights(; hidden=nothing, embed1=nothing, embed2=nothing,
                     vocab1=nothing, vocab2=nothing, atype=nothing, o...)
    nlayer = length(hidden)
    encoder = Array(Any, 2*nlayer+1)
    decoder = Array(Any, 2*nlayer+3)
    input1,input2 = embed1,embed2 # TODO: do we really need embeddings?
    for k = 1:nlayer
        encoder[2k-1] = atype(xavier(input1 + hidden[k], 4*hidden[k]))
        decoder[2k-1] = atype(xavier(input2 + hidden[k], 4*hidden[k]))
        encoder[2k] = atype(zeros(1, 4*hidden[k]))
        decoder[2k] = atype(zeros(1, 4*hidden[k])) # TODO: do we really need biases?
    end
    encoder[2*nlayer+1] = [ atype(xavier(1,embed1)) for i=1:length(vocab1) ]
    decoder[2*nlayer+1] = [ atype(xavier(1,embed2)) for i=1:length(vocab2) ]
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

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="s2s.jl [sequence-to-sequence model](https://arxiv.org/abs/1409.3215). (c) Deniz Yuret, 2017."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="Random number seed.")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="Array and element type.")
        ("--train"; nargs='+'; help="If provided, use first file for training, second for dev, others for eval.")
        ("--hidden"; nargs='+'; arg_type=Int; default=[64]; help="Sizes of one or more LSTM layers.")
        ("--vocab1"; help="Load input vocab (Dict{WordStr,WordIdx}) from file.")
        ("--vocab2"; help="Load input vocab (Dict{WordStr,WordIdx}) from file.")
        ("--embed1"; arg_type=Int; default=512; help="Size of the input embedding vector.")
        ("--embed2"; arg_type=Int; default=32; help="Size of the output embedding vector.")
        ("--epochs"; arg_type=Int; default=5; help="Number of epochs for training.")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        # TODO:
        # ("--test"; help="Apply model to input sequences in test file.")
        # ("--loadfile"; help="Initialize model from file")
        # ("--savefile"; help="Save final model to file")
        # ("--bestfile"; help="Save best model to file")
        # ("--embeddings1"; help="Load input embeddings from file")
        # ("--embeddings2"; help="Load output embeddings from file")
        # ("--batchsize"; arg_type=Int; default=10; help="Size of minibatches.")
        # ("--dropout"; arg_type=Float64; default=0.0; help="Dropout probability.")
        # ("--gclip"; arg_type=Float64; default=0.0; help="Gradient clip.")
        # ("--winit"; arg_type=Float64; default=0.1; help="Stdev of initial random weights.") # using xavier instead
    end
    println(s.description)
    isa(args, AbstractString) && (args=split(args))
    global o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    o[:atype] = eval(parse(o[:atype]))
    o[:vocab1] = o[:vocab1]!=nothing ? load(o[:vocab1], "vocab") : Dict(EOSSTR=>EOS)
    o[:vocab2] = o[:vocab2]!=nothing ? load(o[:vocab2], "vocab") : Dict(EOSSTR=>EOS)
    o[:datas] = [readdata2(file; o...) for file in o[:train]] # yes I know data is already plural :)
    o[:model] = initweights(; o...)
    o[:models] = [ deepcopy(o[:model]) ]
    o[:opt] = oparams(o[:model])
    report(t,p,d)=println((t,[ avgloss(p,di) for di in d ]...))
    !o[:fast] && report(0,o[:model],o[:datas])
    s2sgrad = grad(s2s)
    istate = initstate(o[:model][1])
    for epoch = 1:o[:epochs]
        @time for (x,y) in o[:datas][1]
            o[:grad] = s2sgrad(o[:model],istate,x,y)
            Knet.update!(o[:model],o[:grad],o[:opt])
        end
        push!(o[:models], deepcopy(o[:model]))
        !o[:fast] && report(epoch,o[:model],o[:datas])
    end
    return o
end


# hyperband stuff
include(Knet.dir("examples/hyperband.jl"))
config0 = main("--fast --epochs 0 --train trn tst")
s2sgrad = grad(s2s)

function getconfig2()
    c = Dict()
    c[:hidden] = [ rand(50:200) ] # multi-layer? dropout?, gclip?
    c[:embed1] = rand(100:600)
    c[:embed2] = rand(10:100)
    return c
end

function getloss2(c,n)
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

