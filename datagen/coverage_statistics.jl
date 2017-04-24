using Logging, DataFrames, ProgressMeter, ArgParse

include("data_generator.jl")

function coverage(taskf, unmatched, freq; limit=1e8)
    matched = Set()
    patience = 0
    ind = 0
    patlim = string(taskf) == "lang_only" ? 5e6 : 1e6
    prevtext = ""
    limit = string(taskf) == "turn_and_move_to_x" && limit > 5e5 ? 5e5 : limit
    @showprogress 1 "$(taskf) ... " for i in 1:Int(limit)
        ins, mp = taskf("temp", 1)
        ind += 1

        instext = join(ins.text, " ")
        if instext in unmatched
            push!(matched, instext)
            delete!(unmatched, instext)
            patlim += patience
            patience = 0
        else
            if instext != prevtext
                patience += 1
            end
        end
        prevtext = instext
        if length(unmatched) == 0 || patience >= patlim
            break
        end
    end
    numm = length(matched)
    numu = length(unmatched)
    matchedcount = 0
    for t in matched; matchedcount += freq[t]; end;
    ratio = matchedcount / sum(values(freq))

    info("Task: $taskf")
    info("Total number of generated instances: $ind")

    info("Num of matched: $numm ($matchedcount)")
    for t in matched; info(t); info("Freq: $(freq[t])"); end;
    info("")
    info("Num of unmatched: $numu ($(sum(values(freq)) - matchedcount))")
    for t in unmatched; info(t); info("Freq: $(freq[t])"); end;
    info("")
    info("Coverage of $(taskf): $(100.0 * ratio)%")
    info("Uniq Coverage of $(taskf): $(100.0 * (numm / (numm+numu)))%")
    return ratio
end

function main(args=ARGS)
    tasklist = [turn_to_x, move_to_x, lang_only,
        turn_to_x_and_move, move_to_x_and_turn, turn_to_x_move_to_y,
        move_to_x_turn_to_y, move_until, orient, describe, turn_move_to_x,
        move_turn_to_x, turn_and_move_to_x, turn_move_until,
        turn_to_x_move_until, move_until_turn, move_until_turn_to_x]

    s = ArgParseSettings()
    @add_arg_table s begin
        ("--tasks"; nargs='*'; default=map(string, tasklist))
        ("--lname"; default="coverage.log")
        ("--oname"; default="taskcov.csv")
    end
    o = parse_args(args, s)
    srand(12345)
    Logging.configure(filename=o["lname"])
    Logging.configure(level=INFO)

    grid, jelly, l = getallinstructions(;fname="../data/pickles/databag3.jld")
    alltext = map(ins->join(ins.text, " "), vcat(grid, jelly, l))
    allins = Set(alltext)
    freq = Dict{String, Int}()
    for t in alltext
        if t in keys(freq)
            freq[t] = freq[t] + 1
        else
            freq[t] = 1
        end
    end

    df = DataFrame(TaskName=Any[], Coverage=Any[])

    tasklist = map(x->eval(parse(x)), o["tasks"])

    total = 0.0
    for taskf in tasklist
        ratio = coverage(taskf, copy(allins), freq)
        push!(df, (taskf, ratio*100))
        total += ratio
    end

    info("Total coverage: $(total * 100)%")
    writetable(o["oname"], df)
end

main()
