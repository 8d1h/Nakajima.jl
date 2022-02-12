# this file contains the main algorithm to compute cup products in the
# cohomology ring of S^[n] using the the Lehn--Sorger model
# https://arxiv.org/abs/math/0012166
#
# see also https://arxiv.org/abs/1410.8398
# which is an implementation in Haskell by Kapfer
# 
# note that this implementation contains an error:
# the Euler class of H^*(S,Q) should be -24pt instead of 24pt

==(x::ASnBase, y::ASnBase) = (@assert x.n == y.n; x.x == y.x)
hash(x::ASnBase) = hash(sort(collect(x.x)))

==(x::ASnElem, y::ASnElem) = (@assert x.n == y.n; x.x == y.x)
copy(x::ASnElem{T}) where T = ASnElem(copy(x.x), x.n)
convert(::Type{ASnElem}, x::ASnBase) = ASnElem(Dict([x => 1]), x.n)

function show(io::IO, x::ASnBase)
  for cyc in cycles(x.g)
    s = K3_expr(x.x[cyc])
    print(io, "("*s*")"*join(subscriptify.(cyc), "₋"))
  end
end

show(io::IO, x::ASnElem) = show_via_expressify(io, x)
function expressify(x::ASnElem; context = nothing)
  ans = Expr(:call, :+)
  for (k, m) in x.x
    term = Expr(:call, :*)
    push!(term.args, expressify(m, context = context))
    push!(term.args, string(k))
    push!(ans.args, term)
  end
  ans
end

# multiplication in A{Sn}: for two basis elements
function *(x::ASnBase, y::ASnBase)
  @assert x.n == y.n
  new_orbs = cycles(x.g*y.g)
  ans = Dict([Dict{Vector{Int}, Int}() => 1])
  for orb in orbits(x.g, y.g)
    if length(ans) == 0
      return ASnElem(x.n)
    end
    a = Int[]
    count = [0, 0, 0]
    for cyc in cycles(x.g)
      if cyc[1] in orb
        count[1] += 1
        push!(a, x.x[cyc])
      end
    end
    for cyc in cycles(y.g)
      if cyc[1] in orb
        count[2] += 1
        push!(a, y.x[cyc])
      end
    end
    for cyc in new_orbs
      if cyc[1] in orb
        count[3] += 1
      end
    end
    g = divexact(length(orb) + 2 - sum(count), 2)
    for _ in 1:g
      push!(a, pt)
    end
    a = cup(a)
    if length(a) > 0
      new_ans = typeof(ans)()
      cycs = [cyc for cyc in new_orbs if cyc[1] in orb]
      for (k, m) in ans
        for (i, n) in a
          for (j, l) in kunneth(i, count[3], minus=true)
            ck = copy(k)
            for (cyc, ji) in zip(cycs, j)
              ck[cyc] = ji
            end
            new_ans[ck] = EULER^g*m*n*l
          end
        end
      end
      ans = new_ans
    else
      return ASnElem(x.n)
    end
  end
  return ASnElem(Dict([ASnBase(k, x.n) => m for (k, m) in ans]), x.n)
end

# multiplication in A{Sn}: general case
function *(x::ASnElem, y::ASnElem)
  @assert x.n == y.n
  ans = ASnElem(x.n)
  for (xi, m) in x.x
    for (yi, n) in y.x
      for (k, l) in (xi*yi).x
        add_to!(ans.x, k, m*n*l)
      end
    end
  end
  return ans
end
*(x::ASnElem, y::ASnBase) = x*convert(ASnElem, y)
*(x::ASnBase, y::ASnElem) = convert(ASnElem, x)*y

# a permutation acts on a basis element
function act(g::Perm, x::ASnBase)
  function std(cyc::Vector{Int})
    m, i = findmin(cyc)
    return vcat(cyc[i:end], cyc[1:i-1])
  end
  return ASnBase(Dict([std([g[c] for c in cyc]) => x.x[cyc] for cyc in cycles(x.g)]))
end

function symmetrize(x::ASnBase)
  ans = ASnElem(x.n)
  for g in SymmetricGroup(x.n)
    gx = act(g, x)
    add_to!(ans.x, gx, 1)
  end
  return ans
end

function aSnbase_to_nakajima(x::ASnBase)
  ans = Dict{Int, Vector{Int}}()
  for (k, m) in x.x
    a = length(k)
    if a in keys(ans)
      push!(ans[a], m)
    else
      ans[a] = [m]
    end
  end
  ks = sort(collect(keys(ans)), rev=true)
  return NakajimaOp([k for k in ks for _ in ans[k]], [m for k in ks for m in sort(ans[k], rev=true)])
end

# assuming x is Sn-invariant, decompose it into orbits
function decompose(x::ASnElem{T}) where T
  n = x.n
  ans = Pair{NakajimaOp, T}[]
  xx = copy(x.x)
  while length(xx) > 0
    s = collect(keys(xx))[1]
    ss = symmetrize(s)
    m = divexact(xx[s] * length(ss.x), factorial(n))
    push!(ans, aSnbase_to_nakajima(s) => m)
    for k in keys(ss.x)
      delete!(xx, k)
    end
  end
  return ans
end

function nakajima_to_ASn(q::NakajimaOp)
  ans = Dict{Vector{Int}, Int}()
  i = 1
  for (pp, ll) in zip(q.p, q.l)
    ans[collect(i:i + pp - 1)] = ll
    i += pp
  end
  return symmetrize(ASnBase(ans))
end

# cup product in H^*, using multiplication in A{Sn}
function *(x::CohomClass{T}, y::CohomClass{T}) where T
  @assert x.n == y.n
  ans = Dict{NakajimaOp, T}()
  _ans = [typeof(ans)() for _ in 1:Threads.nthreads()]
  Threads.@threads for ((xi, m), (yi, n)) in [(xim, yin) for xim in x.x for yin in y.x]
    id = Threads.threadid()
    sx = nakajima_to_ASn(xi)
    sy = nakajima_to_ASn(yi)
    for (q, v) in decompose(sx*sy)
      add_to!(_ans[id], q, m*n*v)
    end
  end
  for d in _ans
    for (q, v) in d
      add_to!(ans, q, v)
    end
  end
  return CohomClass(ans, x.n)
end
