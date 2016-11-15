#using ArgParse
include("util.jl")

function main()
	#=
	s = ArgParseSettings()

	@add_arg_table s begin
		"--bs"
		help = "batch size"
		default = 100
		arg_type = Int
	end

	args = parse_args(s)
	=#

	files = ["data/instructions/SingleSentenceZeroInitial.grid.json",
	"data/instructions/SingleSentenceZeroInitial.jelly.json"]
	build_data(files, "data/instructions/SingleSentenceZeroInitial.l.json", string("grid_jelly.jld"); charenc=false)

	files = ["data/instructions/SingleSentenceZeroInitial.grid.json",
	"data/instructions/SingleSentenceZeroInitial.l.json"]
	build_data(files, "data/instructions/SingleSentenceZeroInitial.jelly.json", string("grid_l.jld"); charenc=false)

	files = ["data/instructions/SingleSentenceZeroInitial.l.json",
	"data/instructions/SingleSentenceZeroInitial.jelly.json"]
	build_data(files, "data/instructions/SingleSentenceZeroInitial.grid.json", string("l_jelly.jld"); charenc=false)
end

main()
