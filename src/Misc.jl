function subscriptify(n::Int)
  n >= 0 && return AbstractAlgebra.Generic.subscriptify(n)
  return "₋"*subscriptify(-n)
end

# helper function
# increases the value of key k by v in a dictionary d
function add_to!(d::Dict{T, S}, k::T, v::S) where {T, S}
  if k in keys(d)
    d[k] += v
  else
    d[k] = v
  end
  if d[k] == 0
    delete!(d, k)
  end
  return d
end

###############################################################################
# compute the orbits for the subgroup <f,g>
function orbits(f::Perm, g::Perm)
  n = length(f.d)
  @assert length(g.d) == n
  idx = zeros(Int, n)
  for i in 1:length(cycles(f))
    for x in cycles(f)[i]
      idx[x] = i
    end
  end
  idxx = Dict([i => i for i in 1:length(cycles(f))])
  for i in 1:length(cycles(g))
    orb = idx[cycles(g)[i]]
    m = min(orb...)
    for x in orb
      while idxx[x] > idxx[m]
        x, idxx[x] = idxx[x], idxx[m]
      end
      if idxx[x] < idxx[m]
        idxx[m] = idxx[x]
      end
    end
  end
  orbs = Dict{Int,Vector{Int}}()
  for i in 1:n
    v = idx[i]
    while idxx[v] < v
      v = idxx[v]
    end
    if v in keys(orbs)
      push!(orbs[v], i)
    else
      orbs[v] = [i]
    end
  end
  return sort(collect(values(orbs)))
end

###############################################################################
# symmetric functions: conversion between p and m
# here we compute the matrix of p_to_m using combinatorial methods
# m_to_p is then obtained using the inverse matrix
#
# for a partition p, enumerate all sub-partitions
function sub_part(p::Vector{Int}; rev=true)
  ans = Vector{Tuple}[]
  function enum(part, i)
    i > length(p) && (push!(ans, part[:]); return)
    for j in 1:length(part)
      pj = part[j]
      pjj = tuple(sort(vcat(collect(pj), [p[i]]), rev=rev)...)
      part[j] = pjj
      enum(part, i + 1)
      part[j] = pj
    end
    push!(part, tuple(p[i]))
    enum(part, i + 1)
    pop!(part)
    return
  end
  enum(Tuple[], 1)
  return ans
end

function to_exp_dict(p::Vector{T}) where T
  c = Dict{T, Int}()
  for pp in p
    add_to!(c, pp, 1)
  end
  return c
end

auto(p::Vector) = prod(factorial(mi) for (i, mi) in to_exp_dict(p); init=1)
moment(n::Int, p::Vector) = sum(i^n*mi for (i, mi) in to_exp_dict(p); init=0)

@memoize function p_to_m(p::Vector{Int})
  ans = Dict{Vector{Int}, Int}()
  for pp in sub_part(p)
    pp = sort(sum.(pp), rev=true)
    m = auto(pp)
    add_to!(ans, pp, m)
  end
  ans
end

@memoize function inv_mat(n::Int)
  v = [p.part for p in partitions(n)]
  l = length(v)
  PM = p_to_m.(v)
  M = matrix(QQ, Matrix([v[j] in keys(PM[i]) ? PM[i][v[j]] : 0 for i in 1:l, j in 1:l]))
  iM = inv(M)
  ans = Dict([m => Dict([p => iM[i, j] for (j, p) in enumerate(v) if iM[i, j] != 0]) for (i, m) in enumerate(v)])
end

m_to_p(m::Vector{Int}) = inv_mat(sum(m))[m]

###############################################################################
# enumerate partitions of length d with zero sum
# such that the sum of positive terms is bounded by n
function partitions_with_zero_sum(d::Int, n::Int)
  ans = Vector{Int}[]
  d == 0 && return [Int[]]
  d == 1 && return ans
  function enum(part, i, l, m)
    i == d && (push!(ans, vcat([-sum(part)], part)); return)
    for j in l:m
      if j < 0
        pushfirst!(part, j)
        s = sum(part)
        enum(part, i+1, max(j, -n-s), -s÷(d-i))
        popfirst!(part)
      elseif j > 0
        pushfirst!(part, j)
        s = sum(part)
        enum(part, i+1, j, -s÷(d-i))
        popfirst!(part)
      end
    end
  end
  enum(Int[], 1, -n, -1)
  return ans
end
