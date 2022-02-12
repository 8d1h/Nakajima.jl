# this contains the data for the cup product on a K3 surface S, as well as the
# function to compute the Künneth decomposition of the diagonal Δ in S×S
#
# the basis are numbered following HilbK3 by Kapfer:
# 0:             1_S
# 1-2, 3-4, 5-6: three hyperbolic planes
# 7-14, 15-22:   two E8(-1)
# 23:            point class
#
# TODO optimize the structure to allow other surfaces

if !isdefined(Nakajima, :E8)
const E8 = Matrix([
  -2  1  0  0  0  0  0  0;
   1 -2  1  0  0  0  0  0;
   0  1 -2  1  0  0  0  0;
   0  0  1 -2  1  0  0  0;
   0  0  0  1 -2  1  1  0;
   0  0  0  0  1 -2  0  1;
   0  0  0  0  1  0 -2  0;
   0  0  0  0  0  1  0 -2])
const E8inv = Matrix([
  -2  -3  -4  -5  -6  -4  -3  -2;
  -3  -6  -8 -10 -12  -8  -6  -4;
  -4  -8 -12 -15 -18 -12  -9  -6;
  -5 -10 -15 -20 -24 -16 -12  -8;
  -6 -12 -18 -24 -30 -20 -15 -10;
  -4  -8 -12 -16 -20 -14 -10  -7;
  -3  -6  -9 -12 -15 -10  -8  -5;
  -2  -4  -6  -8 -10  -7  -5  -4])
const pt = 23 # the index for the point class
const EULER = -24
end

function K3_expr(i::Int)
  i == 0 && return "1"
  i == pt && return "pt"
  return "e"*subscriptify(i)
end

function K3_degree(i::Int)
  i == 0 && return 0
  i == pt && return 2
  return 1
end

function K3_bil(i::Int, j::Int)
  i > j && return K3_bil(j, i)
  i == 0 && return j == 23 ? 1 : 0
  i == 1 && return j == 2 ? 1 : 0
  i == 2 && return j == 1 ? 1 : 0
  i == 3 && return j == 4 ? 1 : 0
  i == 4 && return j == 3 ? 1 : 0
  i == 5 && return j == 6 ? 1 : 0
  i == 6 && return j == 5 ? 1 : 0
  i < 15 && return j < 15 ? E8[i - 6, j - 6] : 0
  i < 23 && return j < 23 ? E8[i - 14, j - 14] : 0
  return 0
end

function dual_basis(n::Int)
  n == 0 && return Dict([23 => 1])
  n == 1 && return Dict([2 => 1])
  n == 2 && return Dict([1 => 1])
  n == 3 && return Dict([4 => 1])
  n == 4 && return Dict([3 => 1])
  n == 5 && return Dict([6 => 1])
  n == 6 && return Dict([5 => 1])
  n < 15 && return Dict([i => E8inv[i-6, n-6] for i in 7:14])
  n < 23 && return Dict([i => E8inv[i-14, n-14] for i in 15:22])
  n == 23 && return Dict([0 => 1])
end

@memoize function cup(i::Int, j::Int)::Dict{Int, Int}
  i > j && return cup(j, i)
  i == 0 && return Dict([j => 1])
  i < 23 && j < 23 && (b = K3_bil(i, j); return Dict(b != 0 ? [23 => b] : []))
  return Dict()
end

function cup(x::Dict{Int, Int}, y)
  ans = Dict{Int, Int}()
  for (i, m) in x
    for (k, n) in cup(y, i)
      add_to!(ans, k, m*n)
    end
  end
  return ans
end
cup(x::Int, y::Dict{Int, Int}) = cup(y, x)
cup(list...) = reduce(cup, list; init=Dict([0 => 1]))
function cup(xs::Vector{Int})::Dict{Int, Int}
  length(xs) == 0 && return Dict([0 => 1])
  k, m = 0, 1
  for x in xs
    (k > 0 && x == 23 || k == 23 && x > 0) && return Dict()
    if k > 0 && k < 23 && x > 0 && x < 23
      m *= K3_bil(k, x)
      m == 0 && return Dict()
      k = 23
    end
    if k == 0 k = x end
  end
  return Dict([k => m])
end

# this gives the Künneth decomposition of the diagonal
# the minus version is used for the algebra A{Sn}
@memoize function kunneth(k::Int, n::Int=2; minus=false)
  ans = Dict{Tuple, Int}()
  n == 0 && return k == pt ? Dict([Tuple([]) => 1]) : ans
  n == 1 && return Dict([(k,) => 1])
  if n == 2
    for i in 0:23
      for j in 0:23
        aij = cup(dual_basis(i), dual_basis(j), k)
        if pt in keys(aij) && aij[pt] != 0
          ans[(i,j)] = minus ? -aij[pt] : aij[pt]
        end
      end
    end
    return ans
  end
  for ((i, j), m) in kunneth(k, minus=minus)
    for (rest, l) in kunneth(j, n - 1, minus=minus)
      ans[Tuple(vcat([i], collect(rest)))] = m*l
    end
  end
  return ans
end
