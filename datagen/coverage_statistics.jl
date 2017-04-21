using Logging, DataFrames, ProgressMeter

include("data_generator.jl")

function coverage(taskf, unmatched; limit=1e6)
    matched = Set()
    patience = 0
    ind = 0
    @showprogress 1 "$(taskf) ... " for i in 1:Int(limit)
        ins, mp = taskf("temp", 1)
        ind += 1

        instext = join(ins.text, " ")
        if instext in unmatched
            push!(matched, instext)
            delete!(unmatched, instext)
            patience = 0
        else
            patience += 1
        end
        if length(unmatched) == 0 || patience >= 20000
            break
        end
    end
    numm = length(matched)
    numu = length(unmatched)
    ratio = numm / (numm + numu)

    info("Task: $taskf")
    info("Total number of generated instance: $ind")

    info("Num of matched: $numm")
    for t in matched; info(t); end;
    info("")
    info("Num of unmatched: $numu")
    for t in unmatched; info(t); end;
    info("")
    info("Coverage of $(taskf): $(100.0 * ratio)%")
    return ratio
end

function main()
    srand(12345)
    Logging.configure(filename="coverage.log")
    Logging.configure(level=INFO)

    grid, jelly, l = getallinstructions(;fname="../data/pickles/databag3.jld")
    allins = Set(map(ins->join(ins.text, " "), vcat(grid, jelly, l)))
    df = DataFrame(TaskName=Any[], Coverage=Any[])
    tasklist = [turn_to_x, move_to_x, turn_and_move_to_x, lang_only,
        turn_to_x_and_move, move_to_x_and_turn, turn_to_x_move_to_y,
        move_to_x_turn_to_y, move_until, orient, describe]

    for taskf in tasklist
        ratio = coverage(taskf, copy(allins))
        push!(df, (taskf, ratio*100))
    end
    writetable("taskcov.csv", df)
end

main()
