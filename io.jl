using JLD

function savemodel(w, fname; flex=false)
    d = Dict()
    for k in keys(w)
        if flex && startswith(k, "filter")
            d[k] = map(x->convert(Array, x), w[k])
        else
            d[k] = convert(Array, w[k])
        end
    end
    save(fname, "weights", d)
end

function loadmodel(fname; flex=false)
    w = Dict()
    d = load(fname, "weights")
    for k in keys(d)
        if flex && startswith(k, "filter")
            w[k] = map(x->convert(KnetArray, x), d[k])
        else
            w[k] = convert(KnetArray, d[k])
        end
    end
    return w
end
