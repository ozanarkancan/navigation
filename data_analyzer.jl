include("util.jl")

function analyze()
	gname = "data/instructions/SingleSentence.grid.json"
	jname = "data/instructions/SingleSentence.jelly.json"
	lname = "data/instructions/SingleSentence.l.json"
	g_ins = getinstructions(gname)
	j_ins = getinstructions(jname)
	l_ins = getinstructions(lname)


	println("Number of instrictions (G, Jelly, L): $((length(g_ins), length(j_ins), length(l_ins)))")

	vocab = build_dict(vcat(g_ins, j_ins, l_ins))
	println("\nVocab Size: $(length(vocab))")

	vocab1 = Set(keys(build_dict(g_ins)))
	vocab2 = Set(keys(build_dict(j_ins)))
	vocab3 = Set(keys(build_dict(l_ins)))

	println("\nLeave One Out")
	println("Grid-Jelly, Intersect with L: $(length(union(vocab1, vocab2))) $(length(intersect(union(vocab1, vocab2), vocab3)))")
	println("Grid-L, Intersect with Jelly: $(length(union(vocab1, vocab3))) $(length(intersect(union(vocab1, vocab3), vocab2)))")
	println("Jelly-L, Intersect with Grid: $(length(union(vocab2, vocab3))) $(length(intersect(union(vocab2, vocab3), vocab1)))")
end

analyze()
