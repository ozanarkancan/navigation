include("navimap_utils.jl")

@enum Category visual_t visual_m visual_tm langonly_t langonly_m orient_t condition_m description

opposites = Dict(0=>"south", 90=>"west", 180=>"north", 270=>"east")
rights = Dict(0=>"east", 90=>"south", 180=>"west", 270=>"north")
lefts = Dict(0=>"west", 90=>"north", 180=>"east", 270=>"south")
ordinals = Dict(1=>"first", 2=>"second", 3=>"third", 4=>"fourth", 5=>"fifth", 
    6=>"sixth", 7=>"seventh", 8=>"eighth", 9=>"ninth")
times = Dict(1=>"once", 2=>"twice")
numbers = Dict(1=>["one", "a"],2=>["two"],3=>["three"],4=>["four"],5=>["five"],
    6=>["six"],7=>["seven"],8=>["eight"],9=>["nine"],10=>["ten"])
wall_names = Dict(1=>"butterflies",2=>"fish",3=>"towers")
floor_names = Dict(1=>["octagon", "blue-tiled"],2=>["brick"],3=>["bare concrete", "concrete", "plain cement"],
    4=>["flower", "flowered", "pink-flowered", "rose"], 5=>["grass", "grassy"],6=>["gravel", "stone"],
    7=>["wood", "wooden"],8=>["yellow", "yellow-tiled", "honeycomb yellow"])
item_names = Dict(1=>["stool"], 2=>["chair"], 3=>["easel"], 4=>["hatrack", "hat rack", "coatrack", "coat rack"],
    5=>["lamp"], 6=>["sofa", "bench"])

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

function generate_lang(navimap, maze, segments; combine=0.6)
    generation = Any[]

    if length(segments) > 1
        append!(generation, startins(navimap, maze, segments[1], segments[2]))
    end

    ind = 2
    while ind < length(segments)
        if rand() >= combine || ind+2 >= length(segments)
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
            push!(cands, (string("turn ", d), langonly_t))
        else
            push!(cands, ("turn around", langonly_t))
            dir = opposites[curr_s[1][3]]
        end

        diff_w, diff_f = around_different_walls_floor(navimap, (curr_s[1][1], curr_s[1][2]))
        wpatrn, fpatrn = navimap.edges[(next_s[1][1], next_s[1][2])][(next_s[2][1], next_s[2][2])]

        if diff_w
            for prefx in ["look for the ", "face the ", "turn your face to the ", "turn until you see the ", "turn to the "]
                for cor in ["corridor ", "hall ", "alley ", "hallway "]
                    for sufx in ["", " on the wall", " on both sides of the walls"]
                        push!(cands, (string(prefx, cor, "with the ", wall_names[wpatrn], sufx), visual_t))
                    end
                end
            end
        end

        if diff_f
            for prefx in ["look for the ", "face the ", "turn your face to the ", "turn until you see the ", "turn to the "]
                for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                    cors = [" path", " hall", " hallway", " alley", " corridor"]
                    if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered"
                        push!(cors, " carpet")
                    end
                    for cor in cors
                        push!(cands, (string(prefx, flr, cor), visual_t))
                    end
                end
            end

            for verb in ["facing the ", "seeing the "]
                for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                    cors = [" path", " hall", " hallway", " alley", " corridor"]
                    if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered"
                        push!(cors, " carpet")
                    end
                    for cor in cors
                        push!(cands, (string("you should be ", verb, flr, cor), visual_t))
                    end
                end
            end
        end
        
        item = find_single_item_in_visible(navimap, curr_s[1][1:2], next_s[2])
        if item != 7
            for prefx in ["at this intersection ", ""]
                for body in ["look for the ", "face the ", "face the intersection containing the ", "turn your face to the ", "turn until you see the ", "turn to the "]
                    push!(cands, (string(prefx, body, rand(item_names[item])), visual_t))
                end
            end
        end

        if is_deadend(maze, p1)
            push!(cands, ("you should leave the dead end", visual_t))
            for w in ["way ", "direction "]
                for g in ["go", "move", "travel"]
                    push!(cands, (string("only one ", w, "to ", g), visual_t))
                end
            end
        end

        if sum(maze[p1[1], p1[2], :]) == 3 || sum(maze[p1[1], p1[2], :]) == 2
            p = (curr_s[end][2], curr_s[end][1], round(Int, 1+curr_s[end][3] / 90))
            rightwall = maze[p[1], p[2], rightof(p[3])] == 0
            leftwall = maze[p[1], p[2], leftof(p[3])] == 0
            backwall = maze[p[1], p[2], backof(p[3])] == 0

            if rightwall && !backwall && !leftwall
                push!(cands, ("turn so that the wall is on your right", orient_t))
                push!(cands, ("turn so that the wall is on your right side", orient_t))
            elseif rightwall && backwall && !leftwall
                push!(cands, ("turn so that the wall is on your right and back", orient_t))
                push!(cands, ("turn so that the wall is on your back and right", orient_t))
            elseif !rightwall && !backwall && leftwall
                push!(cands, ("turn so that the wall is on your left", orient_t))
                push!(cands, ("turn so that the wall is on your left side", orient_t))
            elseif !rightwall && backwall && leftwall
                push!(cands, ("turn so that the wall is on your left and back", orient_t))
                push!(cands, ("turn so that the wall is on your back and left", orient_t))
            elseif !rightwall && backwall && !leftwall
                push!(cands, ("turn so that your back is to the wall", orient_t))
                push!(cands, ("turn so that your back faces the wall", orient_t))
                push!(cands, ("turn so that your back side faces the wall", orient_t))
                for r in [" to", " against"]
                    push!(cands, (string("place your back", r, " the wall"), orient_t))
                end
            end
        end

        return  [(curr_s, rand(cands))]
    else
        diff_w, diff_f = around_different_walls_floor(navimap, (curr_s[1][1], curr_s[1][2]))
        wpatrn, fpatrn = navimap.edges[(curr_s[1][1], curr_s[1][2])][(curr_s[2][1], curr_s[2][2])]
        l = Any[]

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
        for m in ["forward ", "straight ", ""]
            for st in sts
                for num in numbers[steps]
                    push!(cands, (string(g, m, num, st), langonly_m))
                end
            end
        end
    end

    for v in ["take ", ""]
        for num in numbers[steps]
            for st in sts
                push!(cands, (string(v, num, st), langonly_m))
            end
        end
    end

    if facing_wall(maze, (endpoint[2], endpoint[1], d))
        for cor in [" path", " hall", " hallway", " alley", " corridor"]
            for unt in [" until ", " until you get to ", " until you reach "]
                push!(cands, (string("take the", cor, unt, "the wall"), visual_m))
            end
        end

        for m in ["move ", "go ", "walk "]
            for adv in ["forwards ", "straight ", ""]
                for unt in ["until ", "until you get to ", " until you reach "]
                    push!(cands, (string(m, adv, unt, "the wall"), visual_m))
                end
            end
        end
    end

    p1 = (curr_s[1][2], curr_s[1][1], -1)
    p2 = (curr_s[end][2], curr_s[end][1], -1)

    if is_corner(maze, p2)
        for m in ["move ", "go ", "walk "]
            for adv in [" forward", " straight", ""]
                for prep in [" into the corner", " to the corner"]
                    for suffix in ["", " you see in front of you"]
                        push!(cands, (string(m, adv, prep, suffix), visual_m))
                        for st in sts
                            for num in numbers[steps]
                                push!(cands, (string(m, adv, num, st, prep, suffix), visual_m))
                            end
                        end
                    end
                end
            end
        end
    end

    if is_deadend(maze, p2)
        for m in ["move ", "go ", "walk "]
            for adv in ["forward", "straight", ""]
                push!(cands, (string(m, adv, " into the dead end"), visual_m))
            end
        end
    end

    if (is_corner(maze, p1) || is_deadend(maze, p1)) && (is_corner(maze, p2) || is_deadend(maze, p2))
        for m in ["move", "go", "walk"]
            for sufx in ["hall", "hallway", "path", "corridor", "alley", ""]
                if sufx != ""
                    push!(cands, (string(m, " to the other end of the ", sufx), visual_m))
                else
                    push!(cands, (string(m, " to the other end"), visual_m))
                end
            end
        end
    elseif (is_corner(maze, p2) || is_deadend(maze, p2))
        for m in ["move", "go", "walk"]
            for cor in ["hall", "hallway", "path", "corridor", "alley"]
                for sufx in ["hall", "hallway", "path", "corridor", "alley", ""]
                    if sufx != ""
                        push!(cands, (string(m, " all the way to the end of the ", sufx), visual_m))
                        push!(cands, (string(m, " to the end of the ", sufx), visual_m))
                    else
                        push!(cands, (string(m, " to the end"), visual_m))
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
                        push!(cands, (string(m, cond, "next alley"), condition_m))
                    else
                        push!(cands, (string(m, cond, ordinals[alleycnt], " alley"), condition_m))
                    end
                end
            end
        end
    end

    if navimap.nodes[curr_s[end][1:2]] != 7 && item_single_in_visible(navimap, navimap.nodes[curr_s[end][1:2]], curr_s[1][1:2])
        for m in ["go ", "move ", "walk "]
            for adv in ["forward ", "straight ", ""]
                for num in numbers[steps]
                    for st in sts
                        for tow in [" to", " towards"]
                            push!(cands, (string(m, adv, num, st, tow, " the intersection containing the ",
                               rand(item_names[navimap.nodes[curr_s[end][1:2]]])), visual_m))
                        end
                    end
                end
            end
        end

        for v in ["take ", ""]
            for num in numbers[steps]
                for st in sts
                    for tow in [" to", " towards"]
                        push!(cands, (string(v, num, st, tow, " the intersection containing the ",
                            rand(item_names[navimap.nodes[curr_s[end][1:2]]])), visual_m))
                    end
                end
            end
        end

        for m in ["go ", "move ", "walk "]
            for adv in ["forward ", "straight ", ""]
                for cond in ["until the ", "towards the ", "until you get to ", "until you reach the ", "till you get to a "]
                    push!(cands, (string(m, adv, cond, rand(item_names[navimap.nodes[curr_s[end][1:2]]])), visual_m))
                end
            end
        end

        for cor in ["path", "hall", "hallway"]
            push!(cands, (string("take the ", cor, " towards the ", rand(item_names[navimap.nodes[curr_s[end][1:2]]])), visual_m))
        end

        wpatrn, fpatrn = navimap.edges[(curr_s[1][1], curr_s[1][2])][(curr_s[2][1], curr_s[2][2])]
        for v in ["follow the ", "along the "]
            for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                cors = [" path", " hall", " hallway", " alley", " corridor"]
                if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered"
                    push!(cors, " carpet")
                end
                for cor in cors
                    push!(cands, (string(v, flr, cor, " to the ", rand(item_names[navimap.nodes[curr_s[end][1:2]]])), visual_m))
                end
            end
        end
    elseif navimap.nodes[curr_s[end][1:2]] == 7 && navimap.nodes[curr_s[end-1][1:2]] != 7 && 
        length(curr_s) > 2 && item_single_in_visible(navimap, navimap.nodes[curr_s[end-1][1:2]], curr_s[1][1:2])
        for m in ["move ", "go ", "walk "]
            for one in ["a", "one"]
                for step in [" step", " block", " segment"]
                    push!(cands, (string(m, one, step, " beyond the ",
                        rand(item_names[navimap.nodes[curr_s[end-1][1:2]]])), condition_m))
                end
            end
        end

        push!(cands, (string("one block pass the ", rand(item_names[navimap.nodes[curr_s[end-1][1:2]]])), condition_m))
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
                        if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            for d in [" on your right"]
                                push!(cands, (string(m, adv, "until you see the ",
                                    flr, cor, d), condition_m))
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
                        if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            for d in [" on your left"]
                                push!(cands, (string(m, adv, "until you see the ",
                                    flr, cor, d), condition_m))
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
                        if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            push!(cands, (string(m, adv, "until you reach the ",
                                flr, cor), condition_m))
                        end
                    end
                end
            end

            for m in ["move ", "go ", "walk "]
                for c in ColorMapping[fpatrn]
                    push!(cands, (string(m, "until you reach the ", c, " intersection"), condition_m))
                end
            end

            for m in ["move ", "go ", "walk "]
                for flr1 in vcat(floor_names[fp], ColorMapping[fp])
                    for flr2 in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        push!(cands, (string(m, "until you reach an intersection with ",
                            flr1, " and ", flr2), condition_m))
                    end
                end
            end

            for v in ["take the ", "follow the "]
                for flr1 in vcat(floor_names[fp], ColorMapping[fp])
                    for flr2 in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        for cor1 in [" path", " hall", " hallway", " alley", " corridor"]
                            for cond in [" to the intersection with the ", " until it crosses the ", " until you end up on the "]
                                for cor2 in [" path", " hall", " hallway", " alley", " corridor"]
                                    push!(cands, (string(v, flr1, cor1, cond, flr2, cor2), condition_m))
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
                            push!(cands, (string("follow this", cor1, cond, flr2, cor2), condition_m))
                        end
                    end
                end
            end
        end
    end

    return [(curr_s, rand(cands))]
end

"""
TODO
"""
function turnins(navimap, maze, curr, next)
    curr_t, curr_s = curr
    next_t, next_s = next

    cands = Any[]
    a = action(curr_s[1], curr_s[2])
    d = a == 2 ? "right" : "left"

    for v in ["turn ", "go ", "turn to the ", "make a ", "take a "]
        push!(cands, (string(v, d), langonly_t))
    end

    if is_corner(maze, (curr_s[1][2], curr_s[1][1], round(Int, curr_s[1][3]/90 + 1)))
        push!(cands, (string("at the corner turn ", d), langonly_t))
        push!(cands, (string("turn ", d, " at the corner"), langonly_t))
    end

    diff_w, diff_f = around_different_walls_floor(navimap, (curr_s[1][1], curr_s[1][2]))
    wpatrn, fpatrn = navimap.edges[(next_s[1][1], next_s[1][2])][(next_s[2][1], next_s[2][2])]

    if diff_w
        for prefx in ["at this intersection ", ""]
            for cor in ["corridor ", "hall ", "alley "]
                for v in ["look for the ", "face the ", "turn your face to the ", "turn to the ", "turn until you see the "]
                    for sufx in ["", " on the wall", " on both sides of the walls"]
                        push!(cands, (string(prefx, v, cor, "with the ", wall_names[wpatrn], sufx), visual_t))
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
                        push!(cands, (string(prefx, v, flr, cor), visual_t))
                    end
                end
            end
        end

        for prefx in ["at this intersection ", ""]
            for cor in [" corridor", " hall", " alley", " hallway", " path"]
                for v in ["facing the ", "seeing the "]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        push!(cands, (string(prefx, "you should be ", v, flr, cor), visual_t))
                    end
                end
            end
        end
    end

    item = find_single_item_in_visible(navimap, curr_s[1][1:2], next_s[2])
    if item != 7
        for prefx in ["at this intersection ", ""]
            for body in ["look for the ", "face the ", "turn your face to the ", "turn until you see the ", "turn to the "]
                push!(cands, (string(prefx, body, rand(item_names[item])), visual_t))
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
        newins = string(mi[1], rand([" and ", " then ", " and then "]), ti[1])
        return [(ms, (newins, mi[2], ti[2]))]
    end
end

function turnmoveins(navimap, maze, curr, next, next2)
    steps = length(next)-1
    cands = Any[]
    curr_t, curr_s = curr
    next_t, next_s = next

    segm = copy(curr_s)
    append!(segm, next_s[2:end])


    if navimap.nodes[next_s[end][1:2]] != 7 && item_single_in_visible(navimap, navimap.nodes[next_s[end][1:2]], curr_s[1][1:2])
        for tv in ["turn and ", "face and "]
            for mv in ["move ", "go ", "walk "]
                push!(cands, (string(tv, mv, rand(["forward ", "straight ", ""]),
                    rand(["to the ", "towards the "]), rand(item_names[navimap.nodes[next_s[end][1:2]]])), visual_tm))
            end
        end
        push!(cands, (string(rand(["move ", "go ", "walk "]), "towards the ", rand(item_names[navimap.nodes[next_s[end][1:2]]])), visual_tm))
        push!(cands, (string("take the ", rand(["path", "hall"])," towards the ", rand(item_names[navimap.nodes[next_s[end][1:2]]])), visual_tm))

        wpatrn, fpatrn = navimap.edges[(next_s[1][1], next_s[1][2])][(next_s[2][1], next_s[2][2])]
        push!(cands, (string(rand(["turn and follow the ", "along the ", "face and follow the "]),
            rand([rand(floor_names[fpatrn]), rand(ColorMapping[fpatrn])]),
            rand([" path", " hall", " hallway", " alley", " corridor"]), " to the ", 
            rand(item_names[navimap.nodes[next_s[end][1:2]]])), visual_tm))
    end

    if length(cands) == 0
        tins = turnins(navimap, maze, curr, next)
        mins = moveins(navimap, maze, next, next2)

        ts, ti = tins[1]
        ms, mi = mins[1]

        append!(ts, ms[2:end])
        newins = string(ti[1], rand([" and ", " then ", " and then "]), mi[1])
        return [(ts, (newins, ti[2], mi[2]))]
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
        push!(cands, (string(lasti[1], " and that is the ", rand(["target ", "final "]), "position"),lasti[2]))
        push!(cands, (string(lasti[1], " and that is the position ", num), lasti[2]))
        push!(cands, (string(lasti[1], " and there should be the position ", num), lasti[2]))

        insl[end] = (lasts, rand(cands))
        return insl
    else
        num = rand([rand(numbers[rand(2:10)]), rand(2:10)])
        push!(cands, (string("that is the ", rand(["target ", "final "]), "position"), description))
        push!(cands, (string("that is the position ", num), description))
        push!(cands, (string("there should be the position ", num), description))
        push!(cands, (string("position ", num, " should be there"), description))

        if navimap.nodes[curr_s[end][1:2]] != 7
            push!(cands, (string(rand(["this intersection contains a ", "there is a ", "there should be a "]),
                rand(item_names[navimap.nodes[curr_s[end][1:2]]])), description))
            push!(cands, ("that's it", description))
            push!(cands, ("and stop", description))
        end

        push!(insl, ([curr_s[end]], rand(cands)))
        return insl
    end
end
