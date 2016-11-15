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
	
	println("Number of examples: $(length(instructions))")
	println("Average instruction length (in chars): $(mean(cs)), std: $(std(cs)), max: $(maximum(cs)), min: $(minimum(cs))")
	println("Average instruction length (in tokens): $(mean(ts)), std: $(std(ts)), max: $(maximum(ts)), min: $(minimum(ts))")
	println("Average path length: $(mean(ps)), std: $(std(ps)), max: $(maximum(ps)), min: $(minimum(ps)), indmax: $(indmax(ps))")

	ind1 = indmin(ps)
	ind2 = indmax(ps)

	println("\nInstruction with shortest path:")
	println(instructions[ind1])
	
	println("\nInstruction with longest path:")
	println(instructions[ind2])

	ind1 = indmin(ts)
	ind2 = indmax(ts)

	println("\nInstruction with shortest text:")
	println(instructions[ind1])
	
	println("\nInstruction with longest text:")
	println(instructions[ind2])

	println("\nTop 10 longest: $(sort(ps; rev=true)[1:10])")
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

	println("\n*** Paragraph ***")
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
