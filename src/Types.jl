# a Nakajima operator is represented by two arrays
struct NakajimaOp
  p::Vector{Int}
  l::Vector{Int}
  function NakajimaOp(p::Vector, l::Vector)
    @assert length(p) == length(l)
    new(p, l)
  end
  NakajimaOp(p::Vector, l::Tuple) = NakajimaOp(p, collect(l))
end

# basis element of A{Sn}
struct ASnBase
  x::Dict{Vector{Int}, Int}
  n::Int
  g::Perm{Int}
  function ASnBase(x::Dict{Vector{Int}, Int}, n::Int=sum(length(cyc) for cyc in keys(x); init=0))
    p = Vector{Int}(undef, n)
    for cyc in keys(x)
      for i in 1:length(cyc) - 1
        p[cyc[i]] = cyc[i + 1]
      end
      p[cyc[end]] = cyc[1]
    end
    return new(x, n, Perm(p))
  end
end

# elements of A{Sn}, expressed as a `Dict`
struct ASnElem{T}
  x::Dict{ASnBase, T}
  n::Int
  function ASnElem(x::Dict{ASnBase, T}, n::Int=-1) where T
    if length(x) > 0
      n = collect(keys(x))[1].n
    end
    new{T}(x, n)
  end
  function ASnElem(n::Int)
    new{Int}(Dict{ASnBase, Int}(), n)
  end
end

abstract type NakajimaAbstractType{T} end

# an expression in Nakajima operators
struct NakajimaExpr{T} <: NakajimaAbstractType{T}
  x::Dict{NakajimaOp, T}
  function NakajimaExpr{T}(x::Dict) where T
    new{T}(x)
  end
  function NakajimaExpr(x::Dict{NakajimaOp, T}) where T
    NakajimaExpr{T}(x)
  end
  function NakajimaExpr{T}() where T
    new{T}(Dict())
  end
end

# an expression in Nakajima operators, truncated at `trunc`
# i.e., if trunc = -n, only q_k with k >= -n will be kept
# this will ignore all unwanted terms when working on a fixed S^[n]
struct NakajimaExprTrunc{T} <: NakajimaAbstractType{T}
  x::Dict{NakajimaOp, T}
  trunc::Int
  function NakajimaExprTrunc{T}(x::Dict, trunc::Int=0) where T
    new{T}(x, trunc)
  end
  function NakajimaExprTrunc(x::Dict{NakajimaOp, T}, trunc::Int=0) where T
    NakajimaExprTrunc{T}(x, trunc)
  end
  function NakajimaExprTrunc{T}(trunc::Int=0) where T
    new{T}(Dict(), trunc)
  end
end

# cohomology classes are expressed as Nakajima operators acted on |0⟩=1_pt
mutable struct CohomClass{T} <: NakajimaAbstractType{T}
  x::Dict{NakajimaOp, T}
  n::Int
  trunc::Int # 0 : no negative operators
  function CohomClass{T}(x::Dict, n::Int) where T
    new{T}(x, n, 0)
  end
  function CohomClass(x::Dict{NakajimaOp, T}, n::Int) where T
    CohomClass{T}(x, n)
  end
  function CohomClass{T}(n::Int) where T
    new{T}(Dict(), n, 0)
  end
end

zero(x::NakajimaExpr{T}) where T = NakajimaExpr{T}()
zero(x::NakajimaExprTrunc{T}) where T = NakajimaExprTrunc{T}(x.trunc)
zero(x::CohomClass{T}) where T = CohomClass{T}(x.n)

copy(q::NakajimaOp) = NakajimaOp(q.p, q.l)
copy(x::NakajimaExpr) = NakajimaExpr(copy(x.x))
copy(x::NakajimaExprTrunc) = NakajimaExprTrunc(copy(x.x), x.trunc)
copy(x::CohomClass) = CohomClass(copy(x.x), x.n)
