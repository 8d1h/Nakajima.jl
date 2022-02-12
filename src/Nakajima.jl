module Nakajima

import Memoize: @memoize
import Base: +, -, *, ^, ==, show, copy, hash, one, zero, convert, length, isless, iterate, getindex, keys
import AbstractAlgebra
import AbstractAlgebra: Perm, cycles, divexact, matrix, SymmetricGroup, expressify, show_via_expressify, FieldElement, RingElement, integral
import AbstractAlgebra.Generic: partitions
import Nemo
import Nemo: QQ, fmpq, bernoulli # use Nemo for faster matrix inversion

export nakajima, std, cup, ASnBase, nakajima_to_ASn, symmetrize, integral_basis, in_basis, mult_ch, ch, mult_chern, chern
export GG, kunneth, mult_sq, mult_td

include("Types.jl")
include("K3.jl")
include("Misc.jl")
include("ASn.jl")

###############################################################################
# `NakajimaOp`: a single Nakajima operators
length(q::NakajimaOp) = length(q.p)
==(x::NakajimaOp, y::NakajimaOp) = x.p == y.p && x.l == y.l
hash(q::NakajimaOp) = hash((q.p, q.l))
isless(x::NakajimaOp, y::NakajimaOp) = isless((x.p, x.l), (y.p, y.l))

function show(io::IO, q::NakajimaOp)
  for (pp, ll) in zip(q.p, q.l)
    print(io, string("q", subscriptify(pp), "(", K3_expr(ll), ")"))
  end
  if length(q) == 0
    print(io, "Id")
  end
end

# product is just concatenation
function *(x::NakajimaOp, y::NakajimaOp)
  return NakajimaOp(vcat(x.p, y.p), vcat(x.l, y.l))
end

one(q::NakajimaOp) = NakajimaOp(Int[], Int[])
^(x::NakajimaOp, n::Int) = Base.power_by_squaring(x, n)

# reduction to normal order :q:
# i.e., larger indices are on the left
# uses bubble sort and Nakajima commutation relations
#
# see Nakajima - Lectures on Hilbert schemes of points on surfaces
std(q::NakajimaOp) = std_internal(q)

# internal version which saves allocation
function std_internal(q::NakajimaOp, m=1, ans::Dict=Dict{NakajimaOp, Int}(), trunc::Tuple{Bool, Int}=(false,0), inplace::Bool=false)
  m == 0 && return ans
  p, l = inplace ? (q.p, q.l) : (copy(q.p), copy(q.l))
  sorted = length(q) <= 0
  while !sorted
    sorted = true
    s = p[end]
    for i in length(q)-1:-1:1
      s += p[i]
      trunc[1] && s < trunc[2] && return ans
      if p[i] < p[i+1]
        sorted = false
        # the commutator
        if p[i] + p[i+1] == 0
          c = K3_bil(l[i], l[i+1])
          if c != 0
            # the commutator is a newly created instance
            # so it can always be sorted in place
            std_internal(NakajimaOp(
              [p[j] for j in 1:length(q) if j != i && j != i+1],
              [l[j] for j in 1:length(q) if j != i && j != i+1]),
              m*p[i]*c, ans, trunc, true)
          end
        end
        p[i], p[i+1] = p[i+1], p[i]
        l[i], l[i+1] = l[i+1], l[i]
      elseif p[i] == p[i+1]
        if l[i] < l[i+1]
          sorted = false # so that `l` is also sorted
          l[i], l[i+1] = l[i+1], l[i]
        end
      end
    end
  end
  if trunc[1]
    # if trunc[1] is set to true, truncate operators that are too negative
    if length(p) == 0 || sum(ki for ki in p if ki < 0) >= trunc[2]
      add_to!(ans, NakajimaOp(p, l), m)
    end
  else
    add_to!(ans, NakajimaOp(p, l), m)
  end
  return ans
end

###############################################################################

nakajima(p::Vector, l::Vector) = NakajimaExpr{Rational}(Dict([NakajimaOp(p, l)=>1]))

function std!(x::NakajimaAbstractType)
  for (k, m) in x.x
    delete!(x.x, k)
    for (l, n) in std(k)
      add_to!(x.x, l, m*n)
    end
  end
  x
end
std(x::NakajimaAbstractType) = std!(copy(x))

# multiplication by scalar
function *(x::NakajimaAbstractType, a::RingElement)
  xa = copy(x)
  for q in keys(xa.x)
    xa.x[q] *= a
  end
  return xa
end
*(a::RingElement, x::NakajimaAbstractType) = x*a
-(x::NakajimaAbstractType) = -1*x

# action of Nakajima operators
function *(q::NakajimaOp, x::S) where S <: NakajimaAbstractType{T} where T
  addmul!(zero(x), q, x)
end

function addmul!(ans::S, q::NakajimaOp, x::S, m=1) where S <: NakajimaAbstractType{T} where T
  trunc = hasfield(S, :trunc) ? (true, x.trunc) : (false, 0)
  _ans = [typeof(ans.x)() for _ in 1:Threads.nthreads()]
  Threads.@threads for (k, v) in collect(x.x)
    id = Threads.threadid()
    std_internal(q*k, m*v, _ans[id], trunc, true)
  end
  for d in _ans
    for (k, v) in d
      add_to!(ans.x, k, v)
    end
  end
  if x isa CohomClass
    ans.n = x.n + sum(q.p)
  end
  return ans
end

function addmul!(ans::S, q::NakajimaExpr{T}, x::S) where S <: NakajimaAbstractType{T} where T
  for (k, m) in q.x
    addmul!(ans, k, x, m)
  end
  return ans
end

function addmul!(ans::S, q::NakajimaExprTrunc{T}, x::S) where S <: NakajimaAbstractType{T} where T
  for (k, m) in q.x
    addmul!(ans, k, x, m)
  end
  return ans
end

function *(q::NakajimaExpr{T}, x::S) where S <: NakajimaAbstractType{T} where T
  addmul!(zero(x), q, x)
end

function *(q::NakajimaExprTrunc{T}, x::S) where S <: NakajimaAbstractType{T} where T
  addmul!(zero(x), q, x)
end

function expressify(x::NakajimaAbstractType; context = nothing)
  ans = Expr(:call, :+)
  for (q, m) in sort(x.x)
    term = Expr(:call, :*)
    push!(term.args, expressify(m, context = context))
    if x isa CohomClass
      if isone(q)
        push!(term.args, "|0⟩")
      else
        push!(term.args, string(q)*"|0⟩")
      end
    else
      push!(term.args, string(q))
    end
    push!(ans.args, term)
  end
  return ans
end
show(io::IO, x::NakajimaAbstractType) = show_via_expressify(io, x)

function one(x::CohomClass{T}) where T <: FieldElement
  q = NakajimaOp(ones(Int, x.n), zeros(Int, x.n))
  CohomClass{T}(Dict([q => 1//factorial(x.n)]), x.n)
end

one(x::NakajimaExpr{T}) where T = NakajimaExpr{T}(Dict([NakajimaOp([], []) => one(T)]))
one(x::NakajimaExprTrunc{T}) where T = NakajimaExprTrunc{T}(Dict([NakajimaOp([], []) => one(T)]), x.trunc)

function ^(x::NakajimaAbstractType, n::Int)
  n == 0 && return one(x)
  prod(repeat([x], n))
end

==(x::CohomClass, y::CohomClass) = (@assert x.n == y.n; std(x).x == std(y).x)

coerce(x::CohomClass, y::CohomClass) = (@assert x.n == y.n; x)
coerce(x::NakajimaExpr, y::NakajimaExpr) = x
coerce(x::NakajimaExpr, y::NakajimaExprTrunc) = y
coerce(x::NakajimaExprTrunc, y::NakajimaExpr) = x
coerce(x::NakajimaExprTrunc, y::NakajimaExprTrunc) = x.trunc >= y.trunc ? x : y

function +(x::NakajimaAbstractType{T}, y::NakajimaAbstractType{T}) where T
  # this way one cannot add incompatible objects, say CohomClass + NakajimaExpr
  ans = zero(coerce(x, y))
  for (k, m) in x.x add_to!(ans.x, k, m) end
  for (k, m) in y.x add_to!(ans.x, k, m) end
  ans
end
-(x::NakajimaAbstractType, y::NakajimaAbstractType) = x + -1*y

function integral(x::CohomClass{T}) where T
  q = NakajimaOp(repeat([1], x.n), repeat([pt], x.n))
  q in keys(x.x) && return x.x[q]
  return zero(T)
end

###############################################################################
# compute an integral basis of H^2k(S^[n], Z), given by Qin-Wang in
# http://arxiv.org/abs/math/0405600

function integral_basis(p::Vector{Int}, l::Vector{Int})
  count = Dict([0 => Int[]])
  for (pp, ll) in zip(p, l)
    if ll in keys(count)
      push!(count[ll], pp)
    else
      count[ll] = [pp]
    end
  end
  # first treat q_l(1)
  p0 = pop!(count, 0)
  c = to_exp_dict(p0)
  if length(p0) > 0
    ans = [(p0, repeat([0], length(p0))) => 1//prod(m^n*factorial(n) for (m, n) in c)]
  else
    ans = [(Int[], Int[]) => 1//1]
  end
  # treat the rest
  for (k, pk) in count
    new_ans = typeof(ans)()
    for ((pp, ll), m) in ans
      # for q_l(pt), keep the same partition
      # otherwise, use `m_to_p` to obtain the new partitions
      for (ppk, n) in (k == pt ? [(pk, 1)] : m_to_p(pk))
        push!(new_ans, (vcat(pp, ppk), vcat(ll, repeat([k], length(ppk)))) => m*n)
      end
    end
    ans = new_ans
  end
  return std!(CohomClass{Rational}(Dict([NakajimaOp(p,l)=>m for ((p,l), m) in ans]), sum(p)))
end

# enumeration of all possible pairs of (p,l) that appears in H^2k(S^[n])
# TODO tidy up
@memoize function all_pl(k::Int, n::Int)
  # B(d) is the maximal index of a class with degree d
  B = d -> d == 2 ? 23 : (
           d == 1 ? 22 : (
           d == 0 ? 0 : -1))
  # i: index, m: max value, d: remaining degree
  function enum(p, part, i, m, d, ans)
    len = length(p)
    i > len && (push!(ans, (p, part[:])); return)
    if i == 1 || p[i] < p[i-1]
      # reset the max possible value for l[i]
      m = B(min(2, d))
    end
    # set the min possible value for l[i]
    l = d <= 2(len-i)   ? 0 : (
        d == 2(len-i)+1 ? 1 : (
        d == 2(len-i)+2 ? 23 : 24))
    for j in l:m
      push!(part, j)
      dj = K3_degree(j)
      new_m = d-dj < dj ? B(d-dj) : j
      enum(p, part, i+1, new_m, d-dj, ans)
      pop!(part)
    end
  end
  ans = Tuple[]
  for p in partitions(n)
    enum(collect(p), Int[], 1, 23, k - n + length(p), ans)
  end
  return [NakajimaOp(p, l) for (p, l) in ans]
end

@memoize function integral_basis(k::Int, n::Int)
  [integral_basis(q.p, q.l) for q in all_pl(k, n)]
end

@memoize function mat_q_to_int(k::Int, n::Int)
  inv(matrix(QQ, [q in keys(x.x) ? x.x[q] : 0//1 for q in all_pl(k, n), x in integral_basis(k, n)]))
end

function in_basis(x::CohomClass, k::Int=cohom_degree(collect(keys(x.x))[1]))
  v = [q in keys(x.x) ? x.x[q] : 0//1 for q in all_pl(k, x.n)]
  mat_q_to_int(k, x.n) * matrix(QQ, length(v), 1, v)
end

# this degree d means q|0⟩ is in H^2d of some S^[n]
function cohom_degree(q::NakajimaOp)
  sum(q.p) + sum(K3_degree.(q.l)) - length(q)
end

###############################################################################
# express the Chern characters of S^[n] in terms of Nakajima operators
#
# the formula for mult_ch is given by Li-Qin-Wang
# http://arxiv.org/abs/math/0111047v2

function GG(k::Int, j::Int, n::Int)
  ans = NakajimaExprTrunc{Rational}(-n)
  for p in partitions_with_zero_sum(k, n)
    for (l, m) in kunneth(j, k)
      add_to!(ans.x, NakajimaOp(p, l), -1//auto(p)*m)
    end
  end
  if j == 0
    for p in partitions_with_zero_sum(k-2, n)
      for (l, m) in kunneth(pt, k-2)
        add_to!(ans.x, NakajimaOp(p, l), moment(2, p)//auto(p)*m)
      end
    end
  end
  return ans
end

function GG(k::Vector{Int}, j::Int, n::Int)
  ans = NakajimaExprTrunc{Rational}(-n)
  for (l, m) in kunneth(j, length(k))
    for (k, v) in prod(GG(ki, li, n) for (ki, li) in zip(k, l); init=one(ans)).x
      add_to!(ans.x, k, m*v)
    end
  end
  return ans
end

@memoize function mult_ch(k::Int, n::Int)
  e12 = 2 # euler(S)/12
  ans = sum((-1)^(j+1)*GG([k+2-j, j], 0, n) for j in 0:k+2)
  if e12 != 0
    ans += e12*sum((-1)^(j+1)*GG(k-j, pt, n)*GG(j, pt, n) for j in 0:k)
    if k == 0 ans += -e12*GG(0, pt, n) end
  end
  ans
end

# express the Chern classes in terms of ch
function _expp(k::Int)
  e = [Dict([Int[] => 1//1])]
  for i in 1:k
    ei = Dict{Vector{Int}, Rational}()
    for j in 1:i
      for (p, m) in e[i-j+1]
        add_to!(ei, sort(vcat(p, [j])), -m//2i*factorial(2j))
      end
    end
    push!(e, ei)
  end
  e[end]
end

@memoize function mult_chern(k::Int, n::Int)
  ans = NakajimaExprTrunc{Rational}(-n)
  isodd(k) && return ans
  for (p, m) in _expp(k÷2)
    ans += m * mult_ch(Tuple(2 .* p), n)
  end
  ans
end

@memoize ch(k::Int, n::Int) = mult_ch(k, n) * integral_basis(0, n)[1]
@memoize chern(k::Int, n::Int) = mult_chern(k, n) * integral_basis(0, n)[1]
@memoize function mult_ch(ks::Tuple, n::Int)
  length(ks) == 0 && return one(NakajimaExprTrunc{Rational}(-n))
  length(ks) == 1 && return mult_ch(ks[1], n)
  mult_ch(ks[1], n) * mult_ch(ks[2:end], n)
end
@memoize function mult_chern(ks::Tuple, n::Int)
  length(ks) == 0 && return one(NakajimaExprTrunc{Rational}(-n))
  length(ks) == 1 && return mult_chern(ks[1], n)
  mult_chern(ks[1], n) * mult_chern(ks[2:end], n)
end

# Todd class
@memoize function mult_td(k::Int, n::Int)
  isodd(k) && return zero(NakajimaExprTrunc{Rational}(-n))
  b = k -> (-1)^(k-1)*Rational(bernoulli(2k))//2k
  return (-1)^(k÷2)*sum(prod(b.(p))//auto(collect(p))*mult_ch(Tuple(2 .* p), n) for p in partitions(k÷2))
end

# square root of Todd class
@memoize function mult_sq(k::Int, n::Int)
  isodd(k) && return zero(NakajimaExprTrunc{Rational}(-n))
  b = k -> (-1)^(k-1)*Rational(bernoulli(2k))//4k
  return (-1)^(k÷2)*sum(prod(b.(p))//auto(collect(p))*mult_ch(Tuple(2 .* p), n) for p in partitions(k÷2))
end

end # module
