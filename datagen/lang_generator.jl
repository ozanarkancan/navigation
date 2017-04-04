include("navimap_utils.jl")

opposites = Dict(0=>"south", 90=>"west", 180=>"north", 270=>"east")
rights = Dict(0=>"east", 90=>"south", 180=>"west", 270=>"north")
lefts = Dict(0=>"west", 90=>"north", 180=>"east", 270=>"south")
ordinals = Dict(1=>"first", 2=>"second", 3=>"third", 4=>"fourth", 5=>"fifth", 
    6=>"sixth", 7=>"seventh", 8=>"eighth", 9=>"ninth")
times = Dict(1=>"once", 2=>"twice")
numbers = Dict(1=>["one", "a"],2=>["two"],3=>["three"],4=>["four"],5=>["five"],
    6=>["six"],7=>["seven"],8=>["eight"],9=>["nine"],10=>["ten"])
wall_names = Dict(1=>"butterflies",2=>"fish",3=>"towers")
floor_names = Dict(1=>["octagon", "blue-tiled"],2=>["brick"],3=>["concrete"],4=>["flower"],
    5=>["grass"],6=>["gravel", "stone"],7=>["wood", "wooden"],8=>["yellow"])
item_names = Dict(1=>"stool", 2=>"chair", 3=>"easel", 4=>"hatrack",
    5=>"lamp", 6=>"sofa")

function action(curr, next)
    a = 0
    if curr[1] != next[1] || curr[2] != next[2]#move
        a = 1
    elseif !(next[3] == 270 && curr[3] == 0) && (next[3] > curr[3] || (next[3] == 0 && curr[3] == 270))#right
        a = 2
    elseif !(next[3] == 0 && curr[3] == 270) && (next[3] < curr[3] || (next[3] == 270 && curr[3] == 0))#left
        a = 3
    else
        a = 4
    end
    return a
end				

function generate_lang(navimap, maze, segments)
    generation = Any[]

    if length(segments) > 1
        append!(generation, startins(navimap, maze, segments[1], segments[2]))
    end

    ind = 2
    while ind < length(segments)
        if rand() < 0.6 || ind+2 >= length(segments)
            g = (segments[ind][1] == "turn" ? turnins : moveins)(navimap, maze, segments[ind], segments[ind+1])
            ind += 1
        else
            g = (segments[ind][1] == "turn" ? turnmoveins : moveturnins)(navimap, maze, segments[ind], segments[ind+1], segments[ind+2])
            ind += 2
        end

        append!(generation, g)
    end
    append!(generation, finalins(navimap, maze, segments[end]))
    return generation
end

function to_string(generation)
    txt = ""
    for (s, ins) in generation
        txt = string(txt, "\n", s, "\n", ins, "\n")
    end
    return txt
end

function startins(navimap, maze, curr, next)
    """
    TODO
    """
    curr_t, curr_s = curr
    next_t, next_s = next

    a = action(curr_s[1], curr_s[2])
    p1 = (curr_s[1][2], curr_s[1][1], -1)

    if curr_t == "turn"
        cands = Any[]
        dir = ""
        d = ""
        if length(curr_s) == 2
            if a == 2#right
                dir = rights[curr_s[1][3]]
                d = "right"
            else#left
                dir = lefts[curr_s[1][3]]
                d = "left"
            end
            push!(cands, string("turn ", d))
        else
            push!(cands, "turn around")
            dir = opposites[curr_s[1][3]]
        end
        push!(cands, string("turn to the ", dir))
        push!(cands, string("orient yourself to the ", dir))
        push!(cands, string("turn to face the ", dir))

        diff_w, diff_f = around_different_walls_floor(navimap, (curr_s[1][1], curr_s[1][2]))
        wpatrn, fpatrn = navimap.edges[(next_s[1][1], next_s[1][2])][(next_s[2][1], next_s[2][2])]

        if diff_w
            for prefx in ["look for the ", "face the ", "turn your face to the ", "turn to the "]
                for cor in ["corridor ", "hall ", "alley ", "hallway "]
                    for sufx in ["", " on the wall"]
                        push!(cands, string(prefx, cor, "with the ", 
                        wall_names[wpatrn], sufx))

                    end
                end
            end
        end

        if diff_f
            for prefx in ["look for the ", "face the ", "turn your face to the ", "turn until you see the ", "turn to the "]
                for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                    cors = [" path", " hall", " hallway", " alley", " corridor"]
                    if flr == "flower" || flr == "octagon"
                        push!(cors, " carpet")
                    end
                    for cor in cors
                        push!(cands, string(prefx, flr, cor))
                    end
                end
            end

            for verb in ["facing the ", "seeing the "]
                for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                    cors = [" path", " hall", " hallway", " alley", " corridor"]
                    if flr == "flower" || flr == "octagon"
                        push!(cors, " carpet")
                    end
                    for cor in cors
                        push!(cands, string("you should be ", verb, flr, cor))
                    end
                end
            end
        end

        if is_deadend(maze, p1)
            push!(cands, "you should leave the dead end")
            for w in ["way ", "direction "]
                for g in ["go", "move", "travel"]
                    push!(cands, string("only one ", w, "to ", g))
                end
            end
        end

        if sum(maze[p1[1], p1[2], :]) == 3 || sum(maze[p1[1], p1[2], :]) == 2
            p = (curr_s[end][2], curr_s[end][1], round(Int, 1+curr_s[end][3] / 90))
            rightwall = maze[p[1], p[2], rightof(p[3])] == 0
            leftwall = maze[p[1], p[2], leftof(p[3])] == 0
            backwall = maze[p[1], p[2], backof(p[3])] == 0

            if rightwall && !backwall && !leftwall
                push!(cands, "turn so that the wall is on your right")
            elseif rightwall && backwall && !leftwall
                push!(cands, "turn so that the wall is on your right and back")
                push!(cands, "turn so that the wall is on your back and right")
            elseif !rightwall && !backwall && leftwall
                push!(cands, "turn so that the wall is on your left")
            elseif !rightwall && backwall && leftwall
                push!(cands, "turn so that the wall is on your left and back")
                push!(cands, "turn so that the wall is on your back and left")
            elseif !rightwall && backwall && !leftwall
                push!(cands, "turn so that your back is to the wall")
                push!(cands, "turn so that your back faces the wall")
                for r in [" to", " against"]
                    push!(cands, string("place your back", r, " the wall"))
                end
            end
        end

        return  [(curr_s, rand(cands))]
    else
        diff_w, diff_f = around_different_walls_floor(navimap, (curr_s[1][1], curr_s[1][2]))
        wpatrn, fpatrn = navimap.edges[(curr_s[1][1], curr_s[1][2])][(curr_s[2][1], curr_s[2][2])]
        l = Any[]

        if diff_f && is_corner(maze, p1)
            cnds = Any[]
            for verb in ["facing the ", "seeing the "]
                for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                    cors = [" path", " hall", " hallway", " alley", " corridor"]
                    if flr == "flower" || flr == "octagon"
                        push!(cors, " carpet")
                    end
                    for cor in cors
                        push!(cnds, string("you should be ", verb, flr, cor))
                    end
                end
            end
        end

        append!(l, moveins(navimap, maze, curr, next))
        return l
    end
end

function moveins(navimap, maze, curr, next)
    curr_t, curr_s = curr
    next_t = next != nothing ? next[1] : nothing
    next_s = next != nothing ? next[2] : nothing

    endpoint = map(x->round(Int, x), curr_s[end])
    d = round(Int, endpoint[3] / 90 + 1)

    cands = Any[]
    steps = length(curr_s)-1

    sts = steps > 1 ? [" steps", " blocks", " segments", " times"] : [" step", " block", " segment"]
    for g in ["go ", "move ", "walk "]
        for m in ["forward ", "straight ", " "]
            for st in sts
                for num in numbers[steps]
                    push!(cands, string(g, m, num, st))
                end
            end
        end
    end

    for v in ["take ", ""]
        for num in numbers[steps]
            for st in sts
                push!(cands, string(v, num, st))
            end
        end
    end

    if facing_wall(maze, (endpoint[2], endpoint[1], d))
        for cor in [" path", " hall", " hallway", " alley", " corridor"]
            push!(cands, string("take the ", cor, " until the wall"))
        end

        for m in ["move ", "go ", "walk "]
            for adv in ["forwards ", "straight ", ""]
                push!(cands, string(m, adv, "until the wall"))
            end
        end
    end

    p1 = (curr_s[1][2], curr_s[1][1], -1)
    p2 = (curr_s[end][2], curr_s[end][1], -1)

    if is_corner(maze, p2)
        for m in ["move ", "go ", "walk "]
            for adv in [" forward", " straight", ""]
                push!(cands, string(m, adv, " into the corner"))
            end
        end
    end

    if is_deadend(maze, p2)
        for m in ["move ", "go ", "walk "]
            for adv in [" forward", " straight", ""]
                push!(cands, string(m, adv, " into the dead end"))
            end
        end
    end

    if (is_corner(maze, p1) || is_deadend(maze, p1)) && (is_corner(maze, p2) || is_deadend(maze, p2))
        for m in ["move", "go", "walk"]
            for cor in ["hall", "hallway", "path", "corridor", "alley"]
                for sufx in ["hall", "hallway", "path", "corridor", "alley", ""]
                    if sufx != ""
                        push!(cands, string(m, " to the other end of the ", sufx))
                    else
                        push!(cands, string(m, " to the other end"))
                    end
                end
            end
        end
    elseif (is_corner(maze, p2) || is_deadend(maze, p2))
        for m in ["move", "go", "walk"]
            for cor in ["hall", "hallway", "path", "corridor", "alley"]
                for sufx in ["hall", "hallway", "path", "corridor", "alley", ""]
                    if sufx != ""
                        push!(cands, string(m, " to the end of the ", sufx))
                    else
                        push!(cands, string(m, " to the end"))
                    end
                end
            end
        end
    end

    if is_intersection(maze, p2)
        alleycnt = count_alleys(maze, curr_s)
        if alleycnt > 0
            for m in ["move", "go", "walk"]
                for cond in [" until the ", " to the "]
                    if alleycnt == 1
                        push!(cands, string(m, cond, "next alley"))
                    else
                        push!(cands, string(m, cond, ordinals[alleycnt], " alley"))
                    end
                end
            end
        end
    end

    if navimap.nodes[curr_s[end][1:2]] != 7
        for m in ["go ", "move ", "walk "]
            for adv in ["forward ", "straight ", " "]
                for num in numbers[steps]
                    for st in sts
                        for tow in [" to", " towards"]
                            push!(cands, string(m, adv, num, st, tow, " the intersection containing the ",
                            item_names[navimap.nodes[curr_s[end][1:2]]]))
                        end
                    end
                end
            end
        end

        for v in ["take", ""]
            for num in numbers[steps]
                for st in sts
                    for tow in [" to", " towards"]
                        push!(cands, string(v, num, st, tow, " the intersection containing the ",
                        item_names[navimap.nodes[curr_s[end][1:2]]]))
                    end
                end
            end
        end
    end

    if navimap.nodes[curr_s[end][1:2]] != 7 && item_single_on_this_segment(navimap, curr_s)
        for m in ["go ", "move ", "walk "]
            for adv in ["forward ", "straight ", " "]
                for cond in ["until the ", "towards the "]
                    push!(cands, string(m, adv, cond, item_names[navimap.nodes[curr_s[end][1:2]]]))
                end
            end
        end

        for cor in ["path", "hall", "hallway"]
            push!(cands, string("take the ", cor, " towards the ", item_names[navimap.nodes[curr_s[end][1:2]]]))
        end

        wpatrn, fpatrn = navimap.edges[(curr_s[1][1], curr_s[1][2])][(curr_s[2][1], curr_s[2][2])]
        for v in ["follow the ", "along the "]
            for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                cors = [" path", " hall", " hallway", " alley", " corridor"]
                if flr == "flower" || flr == "octagon"
                    push!(cors, " carpet")
                end
                for cor in cors
                    push!(cands, string(v, flr, cor, " to the ", item_names[navimap.nodes[curr_s[end][1:2]]]))
                end
            end
        end
    elseif navimap.nodes[curr_s[end][1:2]] == 7 && navimap.nodes[curr_s[end-1][1:2]] != 7 && 
        length(curr_s) > 2 && item_single_on_this_segment(navimap, curr_s[1:end-1])
        for m in ["move ", "go ", "walk "]
            for one in ["a", "one"]
                for step in [" step", " block", " segment"]
                    push!(cands, string(m, one, step, " beyond the ",
                    item_names[navimap.nodes[curr_s[end-1][1:2]]]))
                end
            end
        end

        push!(cands, string("one block pass the ", item_names[navimap.nodes[curr_s[end-1][1:2]]]))
    end

    if steps >= 3 && next != nothing
        wp, fp = navimap.edges[curr_s[1][1:2]][curr_s[2][1:2]]

        target = getlocation(navimap, next_s[2], 1)

        res, fpatrn = is_floor_unique(navimap, maze, curr_s, target)
        if res == 1
            for m in ["move ", "go ", "walk "]
                for adv in ["forward ", "straight ", ""]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        cors = [" path", " hall", " hallway", " alley", " corridor"]
                        if flr == "flower" || flr == "octagon"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            for d in [" to your right", " on your right"]
                                push!(cands, string(m, adv, "until you see the ",
                                flr, cor, d))
                            end
                        end
                    end
                end
            end
        elseif res == 2
            for m in ["move ", "go ", "walk "]
                for adv in ["forward ", "straight ", ""]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        cors = [" path", " hall", " hallway", " alley", " corridor"]
                        if flr == "flower" || flr == "octagon"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            for d in [" to your left", " on your left"]
                                push!(cands, string(m, adv, "until you see the ",
                                flr, cor, d))
                            end
                        end
                    end
                end
            end
        elseif res == 3
            for m in ["move ", "go ", "walk "]
                for adv in ["forward ", "straight ", ""]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        cors = [" path", " hall", " hallway", " alley", " corridor"]
                        if flr == "flower" || flr == "octagon"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            push!(cands, string(m, adv, "until you reach the ",
                            flr, cor))
                        end
                    end
                end
            end

            for m in ["move ", "go ", "walk "]
                for c in ColorMapping[fpatrn]
                    push!(cands, string(m, "until you reach the ", c, " intersection"))
                end
            end

            for m in ["move ", "go ", "walk "]
                for flr1 in vcat(floor_names[fp], ColorMapping[fp])
                    for flr2 in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        push!(cands, string(m, "until you reach an intersection with ",
                        flr1, " and ", flr2))
                    end
                end
            end

            for v in ["take the ", "follow the "]
                for flr1 in vcat(floor_names[fp], ColorMapping[fp])
                    for flr2 in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        for cor1 in [" path", " hall", " hallway", " alley", " corridor"]
                            for cond in [" to the intersection with the ", " until it crosses the ", " until you end up on the "]
                                for cor2 in [" path", " hall", " hallway", " alley", " corridor"]
                                    push!(cands, string(v, flr1, cor1, cond, flr2, cor2))
                                end
                            end
                        end
                    end
                end
            end

            for cor1 in [" path", " hall", " hallway", " alley", " corridor"]
                for cond in [" until you reach the ", " end up on the "]
                    for flr2 in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        for cor2 in [" path", " hall", " hallway", " alley", " corridor"]
                            push!(cands, string("follow this", cor1, cond, flr2, cor2))
                        end
                    end
                end
            end
        end
    end

    return [(curr_s, rand(cands))]
end

function turnins(navimap, maze, curr, next)
    curr_t, curr_s = curr
    next_t, next_s = next

    cands = Any[]
    a = action(curr_s[1], curr_s[2])
    d = a == 2 ? "right" : "left"

    for v in ["turn ", "go ", "turn to the ", "make a ", "take a "]
        push!(cands, string(v, d))
    end

    if is_corner(maze, (curr_s[1][2], curr_s[1][1], round(Int, curr_s[1][3]/90 + 1)))
        push!(cands, string("at the corner turn ", d))
    end

    diff_w, diff_f = around_different_walls_floor(navimap, (curr_s[1][1], curr_s[1][2]))
    wpatrn, fpatrn = navimap.edges[(next_s[1][1], next_s[1][2])][(next_s[2][1], next_s[2][2])]

    if diff_w
        for prefx in ["at this intersection ", ""]
            for cor in ["corridor ", "hall ", "alley "]
                for v in ["look for the ", "face the ", "turn your face to the ", "turn to the ", "turn until you see the "]
                    for sufx in ["", " on the wall"]
                        push!(cands, string(prefx, v, cor, "with the ", wall_names[wpatrn], sufx))
                    end
                end
            end
        end
    end

    if diff_f
        for prefx in ["at this intersection ", ""]
            for cor in [" corridor", " hall", " alley", " hallway", " path"]
                for v in ["look for the ", "face the ", "turn your face to the ", "turn to the ", "turn until you see the "]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        push!(cands, string(prefx, v, flr, cor))
                    end
                end
            end
        end

        for prefx in ["at this intersection ", ""]
            for cor in [" corridor", " hall", " alley", " hallway", " path"]
                for v in ["facing the ", "seeing the "]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        push!(cands, string(prefx, "you should be ", v, flr, cor))
                    end
                end
            end
        end
    end
    return [(curr_s, rand(cands))]
end

function moveturnins(navimap, maze, curr, next, next2)
    steps = length(curr)-1
    cands = Any[]

    if length(cands) == 0
        mins = moveins(navimap, maze, curr, next)
        tins = turnins(navimap, maze, next, next2)

        ts, ti = tins[1]
        ms, mi = mins[1]

        append!(ms, ts[2:end])
        newins = string(mi, rand([" and ", " then ", " and then "]), ti)
        return [(ms, newins)]
    end
end

function turnmoveins(navimap, maze, curr, next, next2)
    steps = length(next)-1
    cands = Any[]
    curr_t, curr_s = curr
    next_t, next_s = next

    segm = copy(curr[2])
    append!(segm, next[2:end])


    if navimap.nodes[next_s[end][1:2]] != 7 && item_single_on_this_segment(navimap, next_s)
        push!(cands, string(rand(["turn and move ", "turn and go ", "turn and walk "]), rand(["forward ", "straight ", ""]),
        rand(["to the ", "towards the "]), item_names[navimap.nodes[next_s[end][1:2]]]))
        push!(cands, string(rand(["move ", "go ", "walk "]), "towards the ", item_names[navimap.nodes[next_s[end][1:2]]]))
        push!(cands, string("take the ", rand(["path", "hall"])," towards the ", item_names[navimap.nodes[next_s[end][1:2]]]))

        wpatrn, fpatrn = navimap.edges[(next_s[1][1], next_s[1][2])][(next_s[2][1], next_s[2][2])]
        push!(cands, string(rand(["turn and follow the ", "along the "]),
        rand([rand(floor_names[fpatrn]), rand(ColorMapping[fpatrn])]),
        rand([" path", " hall", " hallway", " alley", " corridor"]), " to the ", 
        item_names[navimap.nodes[next_s[end][1:2]]]))
    end

    if length(cands) == 0
        tins = turnins(navimap, maze, curr, next)
        mins = moveins(navimap, maze, next, next2)

        ts, ti = tins[1]
        ms, mi = mins[1]

        append!(ts, ms[2:end])
        newins = string(ti, rand([" and ", " then ", " and then "]), mi)
        return [(ts, newins)]
    end

    return [(segm, rand(cands))]
end

function finalins(navimap, maze, curr)
    """
    TODO
    """
    curr_t, curr_s = curr
    cands = Any[]

    lasti = ""

    if curr_t == "turn"
        insl = turnins(navimap, maze, curr, nothing)
    else
        insl = moveins(navimap, maze, curr, nothing)
    end
    lasts, lasti = insl[end]

    r = rand()
    if r < 0.2
        return insl
    elseif r <= 0.6
        num = rand([rand(numbers[rand(2:10)]), rand(2:10)])
        push!(cands, string(lasti, " and that is the ", rand(["target ", "final "]), "position"))
        push!(cands, string(lasti, " and that is the position ", num))
        push!(cands, string(lasti, " and there should be the position ", num))

        insl[end] = (lasts, rand(cands))
        return insl
    else
        num = rand([rand(numbers[rand(2:10)]), rand(2:10)])
        push!(cands, string("that is the ", rand(["target ", "final "]), "position"))
        push!(cands, string("that is the position ", num))
        push!(cands, string("there should be the position ", num))
        push!(cands, string("position ", num, " should be there"))

        if navimap.nodes[curr_s[end][1:2]] != 7
            push!(cands, string(rand(["this intersection contains a ", "there is a ", "there should be a "]),
            item_names[navimap.nodes[curr_s[end][1:2]]]))
            push!(cands, "that's it")
            push!(cands, "and stop")
        end

        push!(insl, ([curr_s[end]], rand(cands)))
        return insl
    end
end
