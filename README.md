# navigation

data/pickles/databag3.jld has the datasets.

## getallinstructions

util.jl loads getallinstructions() returns a (grid, jelly, L0) triple.
Each member of the triple is an array of Instruction's defined in instruction.jl.
Fields:
fname (file name)
text (array of word strings)
path (array of triples with (x,y,angle))
map (Grid, Jelly or L)
id "2-1" means 2nd paragraph, 1st sentence.
The first element of the path is the starting position.
Each consecutive path element corresponds to one primitive action
(move,right,left).  And the last path element is followed by a stop
action.


## getmap(fname)

Loads the maps.  See mainflex.jl:get_maps() for example call.  The fname is one of:
data/maps/map-grid.json
data/maps/map-jelly.json
data/maps/map-l.json

Loads a Map struct.  Defined in map.jl.  name, nodes, edges are fieldnames.
name: name string
nodes: Dict:(x,y)->item-id (1-7)
edges: Dict:(x,y)->Dict2
Dict2: (x,y)->(wall-id,floor-id)

Coordinate system:
(x,y) = (0,0) corresponds to upper left room.
All coordinates correspond to rooms, not edges.
edges: Dict:(x,y)->Dict2
(x,y) room coordinate where nodes gives the room item.
Keys of Dict2 are the rooms we are connected to.
Values of Dict2 are the wall/floor of the connecting edge.

Path->Action Sequence: in util.jl.
function build_dict(instructions): Creates a Dict:wordStr->index from an iterable of instructions.
function build_char_dict(instructions): Creates a Dict:char->index from an iterable of instructions.
function ins_arr(d, ins): d=vocab, ins=Array{WordStr}, constructs one-hot (array of vectors)
function ins_arr_embed(embeds, d, ins): same as above, constructs embedding array. embeds:Dict{WordStr->Vector}
function ins_char_arr(d, ins): Char related fn.
function state_agent_centric(map, loc; vdims = [39 39]): map::Map, loc::(x,y,a), vdims:dims of agent view
  Real maps do not have edges represented, agent state explicitly represents them.
  Array[vdims..., cell]: cell is the id of the feature, which differs from item,floor,wall ids (which are all 1:n).
  We have multi-hot cell values:
  rooms have 0:1 items and room bit.
  edges have 2 wall/floor bits and edge bit.
  unseen/non-walkable places have an indicator bit.
  acts like 2 spatial dims and a channel.
  possibly edges have also color bits.
function state_agent_centric_multihot(map, loc)
  older cell representation: |items|+4*(|items|+|walls|+|floors|).
  not convolutional.  prior work uses this.  represents everything in a direction in multihot regardless of position.
function action(curr, next)
  curr=(x,y,a), next=(x,y,a), returns action [1:4], 1=move, 2=right, 3=left, 4=stop.
function build_instance(instance, map, vocab; vdims=[39, 39], emb=nothing, encoding="grid")
  instance::Instruction, map::Map, vocab::Dict{WordStr,WordId}
  output: (words, states, Y)
  words: one-hot or embedding representation of words (array of vectors).
  states: agent centric state array.
  example: if path has three (x,y,a) triples, state[1] corresponds to the first (x,y,a).
  number of states is the same as number of path elements.
  Y: gold action sequence, number of actions also same as number of path elements with last one being stop.
function minibatch(data; bs=100)
  not used.
function build_data(trn_ins, tst_ins, outfile; charenc=false, encoding="grid")
  not used.

mainflex.jl:
function parse_commandline()
function execute(train_ins, test_ins, maps, vocab, emb, args; dev_ins=nothing)
  train_ins: pair of Array{Instruction} for two maps.
  test_ins: single Array{Instruction} for half of test map.
  maps: Dict{String->Map}
  vocab: Dict{WordStr->WordId}
  emb: Dict{WordStr->Vector}
  args: options
  dev_ins:  single Array{Instruction} for half of test map.

  data: for each Instruction in the two training maps, call build_instance.
  trn_data: reshapes the data.
  train_data: replaces each Instruction with a (Instruction,Embedding) pair.
function get_maps()
function mainflex()

Word embeddings:
  word2vec's 300 dimensional word embeddings for vocab entries in task.
  data/embeddings.jld: Dict{WordStr,Vector}
