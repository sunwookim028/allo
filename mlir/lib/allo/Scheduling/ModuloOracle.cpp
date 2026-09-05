/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The sigma/lap modulo feasibility oracle (declared in Scheduler.h).
//
// Start times decompose as `start = T * lap + sigma`, sigma in [0, T).
// Resource capacity lives entirely in sigma space, decided by a CP-SAT model
// over the contending operations' slots. Given a full sigma assignment every
// dependence becomes a lap difference constraint, checked by one Bellman-Ford
// sweep; a positive lap cycle returns as a region cut over the sigma
// assignments that keep it. INFEASIBLE from the sigma model is an
// unconditional proof.
//===----------------------------------------------------------------------===//

#include "allo/Scheduling/Scheduler.h"

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"

using namespace mlir;
using namespace mlir::allo;

using circt::scheduling::Problem;
using operations_research::Domain;
using operations_research::sat::BoolVar;
using operations_research::sat::CpModelBuilder;
using operations_research::sat::CpModelProto;
using operations_research::sat::CpSolverResponse;
using operations_research::sat::CpSolverStatus;
using operations_research::sat::IntVar;
using operations_research::sat::LinearExpr;
using operations_research::sat::SatParameters;
using operations_research::sat::SolutionIntegerValue;

namespace {

/// Rounds of the propose-check-cut loop before giving up.
constexpr unsigned kOracleRounds = 64;

int64_t ceilDiv(int64_t a, int64_t b) {
  assert(b > 0);
  return a >= 0 ? (a + b - 1) / b : -((-a) / b);
}

/// An edge in the fixed-interval constraint graph, its weight already
/// `latency - T * distance` (valid at one T only).
struct FlatEdge {
  unsigned src, dst;
  int64_t w;
};

} // namespace

ModuloOracleResult
mlir::allo::decideModuloFeasibility(ModuloOccupancyProblem &prob,
                                    ArrayRef<Problem::Dependence> breaks,
                                    unsigned ii, double budget) {
  const int64_t T = ii;
  assert(T >= 1 && "an interval to decide");
  ModuloOracleResult out;

  const auto &ops = prob.getOperations();
  const unsigned n = ops.size();
  DenseMap<Operation *, unsigned> index;
  for (Operation *op : ops)
    index.try_emplace(op, index.size());

  // The graph at this interval, edge weights `latency - T * distance`; a chain
  // break costs one extra cycle, as the simplex engine weighs it.
  std::vector<FlatEdge> raw;
  for (Operation *op : ops)
    for (auto &dep : prob.getDependences(op)) {
      auto src = index.find(dep.getSource());
      if (src == index.end())
        continue;
      int64_t w = prob.separationOf(dep);
      int64_t dist = prob.getDistance(dep).value_or(0);
      raw.push_back({src->second, index.at(op), w - T * dist});
    }
  for (const Problem::Dependence &dep : breaks) {
    auto src = index.find(dep.getSource());
    auto dst = index.find(dep.getDestination());
    if (src == index.end() || dst == index.end())
      continue;
    raw.push_back(
        {src->second, dst->second, prob.latencyOf(dep.getSource()) + 1});
  }

  SmallVector<Operation *> contending;
  std::vector<int> cOf(n, -1); // node -> contending index, -1 free
  for (Operation *op : ops)
    if (prob.holdsLimitedUnit(op)) {
      cOf[index.at(op)] = contending.size();
      contending.push_back(op);
    }
  const unsigned nc = contending.size();
  if (!nc) {
    // Nothing contends: the caller's resource-free SDC bound already decided
    // feasibility.
    out.verdict = ModuloFeasibility::Unknown;
    return out;
  }

  // Longest paths with all-free intermediates: composes free ops away so the
  // lap system runs over contending nodes alone. Source nullopt is the origin
  // (every start is at least zero).
  auto compose = [&](std::optional<unsigned> source,
                     std::vector<int64_t> &dist) -> bool {
    if (source) {
      dist.assign(n, INT64_MIN);
      dist[*source] = 0;
    } else {
      dist.assign(n, 0);
    }
    bool changed = true;
    for (unsigned round = 0; changed && round <= n; ++round) {
      changed = false;
      for (const FlatEdge &e : raw) {
        if (dist[e.src] == INT64_MIN)
          continue;
        if (cOf[e.src] >= 0 && (!source || e.src != *source))
          continue; // only free intermediates carry a composed path
        if (dist[e.src] + e.w > dist[e.dst]) {
          dist[e.dst] = dist[e.src] + e.w;
          changed = true;
        }
      }
    }
    return !changed; // false: a positive all-free cycle
  };

  std::vector<int64_t> originIn;
  if (!compose(std::nullopt, originIn)) {
    // A positive all-free cycle is infeasible at this interval whatever the
    // slots.
    out.verdict = ModuloFeasibility::Infeasible;
    return out;
  }

  // Composed contending-to-contending edges; one max per pair suffices, since
  // at fixed T the lap bound is monotone in the composed weight.
  struct CEdge {
    int src, dst; // contending indices; -1 is the origin
    int64_t w;
  };
  std::vector<CEdge> cedges;
  for (unsigned ci = 0; ci < nc; ++ci) {
    std::vector<int64_t> dist;
    if (!compose(index.at(contending[ci]), dist)) {
      out.verdict = ModuloFeasibility::Infeasible;
      return out;
    }
    for (unsigned v = 0; v < n; ++v) {
      if (cOf[v] < 0 || dist[v] == INT64_MIN)
        continue;
      if (cOf[v] == (int)ci) {
        if (dist[v] > 0) { // a positive recurrence through this node alone
          out.verdict = ModuloFeasibility::Infeasible;
          return out;
        }
        continue;
      }
      cedges.push_back({(int)ci, cOf[v], dist[v]});
    }
  }
  for (unsigned v = 0; v < n; ++v)
    if (cOf[v] >= 0 && originIn[v] > INT64_MIN)
      cedges.push_back({-1, cOf[v], originIn[v]});

  // Rotating every start by the same amount is a symmetry, so one op's slot is
  // pinned to zero. Composed path bounds window every other op's start
  // difference against the pivot, so only the residues that window admits need
  // a slot literal. All-pairs longest paths over the contending graph compose
  // start-space difference bounds transitively; a positive diagonal is
  // infeasible.
  std::vector<int64_t> pathW(nc * nc, INT64_MIN);
  for (const CEdge &e : cedges)
    if (e.src >= 0)
      pathW[e.src * nc + e.dst] = std::max(pathW[e.src * nc + e.dst], e.w);
  for (unsigned k = 0; k < nc; ++k)
    for (unsigned i = 0; i < nc; ++i) {
      int64_t ik = pathW[i * nc + k];
      if (ik == INT64_MIN)
        continue;
      for (unsigned j = 0; j < nc; ++j) {
        int64_t kj = pathW[k * nc + j];
        if (kj != INT64_MIN && ik + kj > pathW[i * nc + j])
          pathW[i * nc + j] = ik + kj;
      }
    }
  for (unsigned i = 0; i < nc; ++i)
    if (pathW[i * nc + i] > 0) {
      out.verdict = ModuloFeasibility::Infeasible;
      return out;
    }
  // The pivot with the most finite pairs anchors the tightest windows.
  unsigned pivot = 0;
  unsigned bestFinite = 0;
  for (unsigned ci = 0; ci < nc; ++ci) {
    unsigned finite = 0;
    for (unsigned cj = 0; cj < nc; ++cj)
      finite += (pathW[ci * nc + cj] != INT64_MIN) +
                (pathW[cj * nc + ci] != INT64_MIN);
    if (finite > bestFinite) {
      bestFinite = finite;
      pivot = ci;
    }
  }
  SmallVector<SmallVector<int64_t>> admissible(nc);
  for (unsigned ci = 0; ci < nc; ++ci) {
    if (ci == pivot) {
      admissible[ci].push_back(0);
      continue;
    }
    int64_t lo = pathW[pivot * nc + ci];
    int64_t hi = pathW[ci * nc + pivot] == INT64_MIN ? INT64_MAX
                                                     : -pathW[ci * nc + pivot];
    if (lo == INT64_MIN || hi == INT64_MAX || hi - lo + 1 >= T) {
      for (int64_t s = 0; s < T; ++s)
        admissible[ci].push_back(s);
      continue;
    }
    for (int64_t d = lo; d <= hi; ++d)
      admissible[ci].push_back(((d % T) + T) % T);
    llvm::sort(admissible[ci]);
    admissible[ci].erase(
        std::unique(admissible[ci].begin(), admissible[ci].end()),
        admissible[ci].end());
  }

  // The sigma model: a slot per contending op, one-hots carrying capacity.
  CpModelBuilder model;
  SmallVector<IntVar> sigma;
  SmallVector<DenseMap<int64_t, BoolVar>> hot(nc);
  for (unsigned ci = 0; ci < nc; ++ci) {
    sigma.push_back(model.NewIntVar(Domain::FromValues(
        std::vector<int64_t>(admissible[ci].begin(), admissible[ci].end()))));
    LinearExpr sum, weighted;
    for (int64_t s : admissible[ci]) {
      BoolVar b = model.NewBoolVar();
      hot[ci].try_emplace(s, b);
      sum += b;
      weighted += LinearExpr::Term(b, s);
    }
    model.AddEquality(sum, 1);
    model.AddEquality(sigma[ci], weighted);
    out.literals += admissible[ci].size();
    if ((int64_t)admissible[ci].size() < T)
      ++out.windowed;
  }
  for (Problem::ResourceType rsrc : prob.getResourceTypes()) {
    unsigned limit = prob.getLimit(rsrc).value_or(0);
    if (!limit)
      continue;
    int64_t constant = 0;
    SmallVector<std::pair<unsigned, int64_t>> windows; // (ci, demand), rem
    SmallVector<int64_t> rems;
    for (unsigned ci = 0; ci < nc; ++ci) {
      Operation *op = contending[ci];
      if (!prob.usesResource(op, rsrc))
        continue;
      int64_t occ = prob.getResourceCycles(op);
      int64_t demand = prob.getResourceDemand(op);
      constant += demand * (occ / T);
      if (occ % T) {
        windows.push_back({ci, demand});
        rems.push_back(occ % T);
      }
    }
    if (constant > (int64_t)limit) {
      out.verdict = ModuloFeasibility::Infeasible;
      return out;
    }
    if (windows.empty())
      continue;
    for (int64_t p = 0; p < T; ++p) {
      LinearExpr used;
      unsigned terms = 0;
      for (auto [wi, win] : llvm::enumerate(windows)) {
        auto [ci, demand] = win;
        for (int64_t k = 0; k < rems[wi]; ++k) {
          auto it = hot[ci].find((p - k + T) % T);
          if (it == hot[ci].end())
            continue; // the window proves this op never covers this slot
          used += LinearExpr::Term(it->second, demand);
          ++terms;
        }
      }
      if (terms)
        model.AddLessOrEqual(used, (int64_t)limit - constant);
    }
  }

  // Propose, check, cut. INFEASIBLE is a proof: every cut only excludes
  // sigma regions a positive lap cycle already proved infeasible.
  SmallVector<int64_t> sig(nc), lap(nc + 1);
  SmallVector<int> parent(nc + 1);
  double &spent = out.spent;
  for (unsigned round = 0; round < kOracleRounds && spent < budget; ++round) {
    out.rounds = round + 1;
    SatParameters params;
    params.set_num_workers(1);
    params.set_random_seed(0);
    params.set_max_deterministic_time(budget - spent);
    // Any sigma satisfying capacity and the cuts is a proposal; nothing is
    // minimized, so the first solution answers the round. Presolve is off
    // since the re-solved model would pay it every round.
    params.set_stop_after_first_solution(true);
    params.set_cp_model_presolve(false);
    const CpModelProto &proto = model.Build();
    CpSolverResponse r =
        operations_research::sat::SolveWithParameters(proto, params);
    spent += r.deterministic_time();
    if (r.status() == CpSolverStatus::INFEASIBLE) {
      out.verdict = ModuloFeasibility::Infeasible;
      return out;
    }
    if (r.status() != CpSolverStatus::OPTIMAL &&
        r.status() != CpSolverStatus::FEASIBLE)
      return out; // Unknown: budget or an aborted solve
    for (unsigned ci = 0; ci < nc; ++ci)
      sig[ci] = SolutionIntegerValue(r, sigma[ci]);

    // The lap system at this sigma: node nc is the origin (sigma 0, lap 0).
    auto lapBound = [&](const CEdge &e) {
      int64_t ds = e.src < 0 ? 0 : sig[e.src];
      int64_t dd = sig[e.dst];
      return ceilDiv(e.w - (dd - ds), T);
    };
    lap.assign(nc + 1, INT64_MIN);
    lap[nc] = 0;
    parent.assign(nc + 1, -1);
    bool grew = true;
    int lastGrown = -1;
    unsigned rounds = 0;
    for (; grew && rounds <= nc + 1; ++rounds) {
      grew = false;
      for (auto [ei, e] : llvm::enumerate(cedges)) {
        int64_t from = e.src < 0 ? lap[nc] : lap[e.src];
        if (from == INT64_MIN)
          continue;
        int64_t reach = from + lapBound(e);
        int64_t &to = lap[e.dst];
        if (reach > to) {
          to = reach;
          parent[e.dst] = ei;
          lastGrown = e.dst;
          grew = true;
        }
      }
    }
    if (!grew) {
      // Feasible: least laps are the ASAP witness. Free operations settle by
      // one more longest-path sweep in start space with contending pinned.
      std::vector<int64_t> start(n, 0);
      for (unsigned ci = 0; ci < nc; ++ci)
        start[index.at(contending[ci])] =
            T * std::max<int64_t>(lap[ci], 0) + sig[ci];
      bool moved = true;
      for (unsigned k = 0; moved && k <= n; ++k) {
        moved = false;
        for (const FlatEdge &e : raw) {
          int64_t reach = start[e.src] + e.w;
          if (reach > start[e.dst]) {
            assert(cOf[e.dst] < 0 &&
                   "the composed lap system missed a contending bound");
            start[e.dst] = reach;
            moved = true;
          }
        }
      }
      assert(!moved && "a positive cycle the lap check admitted");
      for (Operation *op : ops)
        out.starts[op] = (unsigned)start[index.at(op)];
      out.verdict = ModuloFeasibility::Feasible;
      return out;
    }

    // A positive cycle: walk parents from the last-grown node into the
    // cycle, then cut the sigma region that keeps every edge's ceiling. The
    // origin has no incoming edges, so the cycle never passes it.
    int at = lastGrown;
    assert(at >= 0);
    for (unsigned hop = 0; hop <= nc + 1; ++hop) {
      assert(parent[at] >= 0 && "a growing node without a parent");
      at = cedges[parent[at]].src < 0 ? (int)nc : cedges[parent[at]].src;
      assert(at != (int)nc && "the parent walk reached the origin");
    }
    SmallVector<int> cycle;
    int walk = at;
    do {
      assert(parent[walk] >= 0 && "the cycle left the parent forest");
      cycle.push_back(parent[walk]);
      walk = cedges[parent[walk]].src;
      assert(walk >= 0 && "the cycle passed the origin");
    } while (walk != at);
    SmallVector<BoolVar> keeps;
    for (int ei : cycle) {
      const CEdge &e = cedges[ei];
      int64_t K = e.w - T * (lapBound(e) - 1) - 1;
      LinearExpr delta = LinearExpr(sigma[e.dst]);
      if (e.src >= 0)
        delta = delta - sigma[e.src];
      BoolVar keep = model.NewBoolVar();
      model.AddLessOrEqual(delta, K).OnlyEnforceIf(keep);
      model.AddGreaterOrEqual(delta, K + 1).OnlyEnforceIf(keep.Not());
      keeps.push_back(keep);
    }
    SmallVector<BoolVar> nots;
    for (BoolVar k : keeps)
      nots.push_back(k.Not());
    model.AddBoolOr(nots);
  }
  return out; // Unknown
}
