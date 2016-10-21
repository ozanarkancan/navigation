include("instruction.jl")

type Info
	chars
	tokens
	pathlength
end

extract(ins) = map(x -> Info(mapreduce(t -> length(t), +, 0, x.text), length(x.text), length(x.path)), ins)

function printinfo(fname)
	instructions = getinstructions(fname)
	extracted = extract(instructions)

	cs = map(x->x.chars, extracted)
	ts = map(x->x.tokens, extracted)
	ps = map(x->x.pathlength, extracted)

	println("Average instruction length (in chars): $(mean(cs)), std: $(std(cs)), max: $(maximum(cs)), min: $(minimum(cs))")
	println("Average instruction length (in tokens): $(mean(ts)), std: $(std(ts)), max: $(maximum(ts)), min: $(minimum(ts))")
	println("Average path length: $(mean(ps)), std: $(std(ps)), max: $(maximum(ps)), min: $(minimum(ps)), indmax: $(indmax(ps))")
	println(instructions[indmax(ps)].text)
	println(instructions[indmax(ps)].path)
end

function main()
	fname = "data/instructions/SingleSentenceZeroInitial.grid.json"
	println("Grid")
	printinfo(fname)
	
	fname = "data/instructions/SingleSentenceZeroInitial.jelly.json"
	println("\nJelly")
	printinfo(fname)
	
	fname = "data/instructions/SingleSentenceZeroInitial.l.json"
	println("\nL")
	printinfo(fname)

	fname = "data/instructions/ParagraphRandom.grid.json"
	println("\nGrid")
	printinfo(fname)
	
	fname = "data/instructions/ParagraphRandom.jelly.json"
	println("\nJelly")
	printinfo(fname)
	
	fname = "data/instructions/ParagraphRandom.l.json"
	println("\nL")
	printinfo(fname)

end

main()
