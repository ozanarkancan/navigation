using JLD

function savemodel(w, fname; flex=false)
	d = Dict()
	for k in keys(w)
        if startswith(k, "filter") && flex
            d[k] = map(x->convert(Array, x), w[k])
        else
            d[k] = convert(Array, w[k])
        end
    end
	save(fname, "weights", d)
end

function loadmodel(fname)
	w = Dict()
	d = load(fname, "weights")
	for k in keys(d)
        if startswith(k, "filter")
            w[k] = map(x->convert(KnetArray, x), d[k])
        else
            w[k] = convert(KnetArray, d[k])
        end
    end
	return w
end
