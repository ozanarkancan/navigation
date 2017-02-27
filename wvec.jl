module WordVec

using PyCall

global const word2vec = PyCall.pywrap(PyCall.pyimport("gensim.models.word2vec"))

type Wvec
    model
end

function Wvec(fname::String; bin=true)
    Wvec(word2vec.Word2Vec["load_word2vec_format"](fname, binary=bin))
end

function getvec(m::Wvec, word::AbstractString)
    vec = nothing
    unk = false
    try
        vec = m.model["__getitem__"](word)
    catch
        vec = m.model["__getitem__"]("unk")
        unk = true
    end
    return vec, unk
end

export Wvec;
export getvec;

end
