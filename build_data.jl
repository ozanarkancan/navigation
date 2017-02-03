using ArgParse
include("util.jl")

function main()
	s = ArgParseSettings()

	@add_arg_table s begin
		"--encoding"
		help = "encoding (grid or multihot)"
		default = "grid"
	end

	args = parse_args(s)

	grid, jelly, l = getallinstructions()
	t = Instruction[]
	append!(t, grid)
	append!(t, jelly)
	build_data(t, l, "grid_jelly2.jld"; charenc=false, encoding=args["encoding"])

	t2 = Instruction[]
	append!(t2, grid)
	append!(t2, l)
	build_data(t2, jelly, "grid_l2.jld"; charenc=false, encoding=args["encoding"])

	t3 = Instruction[]
	append!(t3, l)
	append!(t3, jelly)
	build_data(t3, grid, "l_jelly2.jld"; charenc=false, encoding=args["encoding"])
end

main()
