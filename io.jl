using JLD

function savemodel(w, fname)
	d = Dict()
	for k in keys(w); d[k] = convert(Array, w[k]); end
	save(fname, "weights", d)
end

function loadmodel(fname)
	w = Dict()
	d = load(fname, "weights")
	for k in keys(d); w[k] = convert(KnetArray, d[k]); end;
	return w
end
