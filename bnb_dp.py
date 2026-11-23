"""
===========================================================================================
 bnb_dp.py  -  Chemotherapy dose scheduling (Level 1): OUR OWN SOLVERS
 Dynamic Programming (DP)  and  LP-based Branch-and-Bound (B&B)
===========================================================================================

THE PROBLEM IN ONE PARAGRAPH
    A tumour of size S0 grows by a factor A every period (week).  At the start of each period
    we may give a dose (X[k] = 1) that removes a fraction RF of the tumour.  We must keep the
    tumour size after dosing between S_MIN and S_TOL in EVERY period, and we want the FEWEST
    doses (each dose costs P).  Writing S[k] for the size after dosing in period k and G[k] for
    the size just before dosing:

        G[1] = S0,   G[k] = A * S[k-1]          (growth)
        S[k] = G[k] * (1 - RF * X[k])           (a dose removes a fraction RF)
        S_MIN <= S[k] <= S_TOL                  (size limits)
        X[k] in {0, 1}                          (a dose is given or not)
        minimise  P * (X[1] + ... + X[K])

    The product G[k]*X[k] multiplies two unknowns, so as written the problem is a MINLP
    (mixed-integer NONLINEAR program).

STEP 1 - THE LOG TRANSFORM (why the problem becomes easy to handle)
    Let y[k] = ln S[k].  Taking the natural log of the two equations above:

        y[1] = ln S0 - lam*X[1]
        y[k] = y[k-1] + ln A - lam*X[k]            for k >= 2,      lam = -ln(1 - RF)

    Now everything is LINEAR in (X, y).  The nonlinear term disappeared and no "big-M"
    trick is needed.  The size limits become  ln S_MIN <= y[k] <= ln S_TOL.

    Unrolling the recursion gives a formula worth remembering:

        y[k] = ln S0 + (k-1)*ln A - lam * n[k],      n[k] = X[1] + ... + X[k]  (doses so far)

    ==> the tumour size after period k depends ONLY on k and on HOW MANY doses n[k] were given
        so far - not on WHEN they were given.  This single fact makes dynamic programming work.

STEP 2 - TWO EXACT ALGORITHMS (both are explained in detail next to their code below)
    * dp_solve        : dynamic programming over the state (k, n).
    * branch_and_bound: solve LP relaxations, branch on fractional doses, prune with bounds.

GLOSSARY (terms you will see in the comments)
    feasible        : satisfies every constraint.
    incumbent       : the best FEASIBLE solution found so far (an upper bound on the optimum,
                      because we are minimising).
    lower bound     : a number the optimum cannot go below (from a relaxation).
    relaxation      : the same problem with some constraints loosened (here: X may be any
                      number in [0, 1] instead of exactly 0 or 1).  It is easier to solve and its
                      optimum is a valid lower bound for the true problem.
    gap             : incumbent - lower bound.  Gap = 0  means the incumbent is PROVEN optimal.
    global optimum  : the best solution over the WHOLE feasible region.
    local optimum   : a solution no neighbour improves on (could still be beaten far away).
===========================================================================================
"""
import heapq          # a "priority queue" (always gives back the smallest item first)
import math
import time
import numpy as np
from scipy.optimize import linprog        # LP solver (HiGHS).  We only use it for LP relaxations.
from scipy.sparse import lil_matrix       # sparse matrices: most constraint coefficients are 0

# Default parameters = the paper's example instance (fictitious drug).
#   S0 initial tumour size, A growth factor per period, RF fraction removed by a dose,
#   S_TOL maximum allowed size, S_MIN minimum allowed size after a dose, P cost of one dose.
DEFAULT = dict(S0=50.0, A=1.5, RF=0.6, S_TOL=500.0, S_MIN=10.0, P=10.0)


# =========================================================================================
# PART 1.  THE ORIGINAL (NONLINEAR) MODEL - used to CHECK every answer we produce
# =========================================================================================
def simulate(X, p=DEFAULT):
    """Run the ORIGINAL model for a 0/1 schedule X and return the tumour sizes S[k] after dosing.
    We deliberately do NOT use the log transform here, so this is an independent check."""
    S = []
    for k, x in enumerate(X):
        # size before dosing: the start value in period 1, otherwise last period's size grown by A
        G = p["S0"] if k == 0 else p["A"] * S[-1]
        # a dose (x = 1) removes the fraction RF; no dose (x = 0) leaves the tumour unchanged
        S.append(G * (1 - p["RF"] * x))
    return np.array(S)


def feasible(X, p=DEFAULT):
    """True if schedule X keeps S_MIN <= S[k] <= S_TOL in every period (tiny tolerance for rounding)."""
    S = simulate(X, p)
    return bool(S.max() <= p["S_TOL"] * (1 + 1e-9) and S.min() >= p["S_MIN"] * (1 - 1e-9))


def heuristic(K, p=DEFAULT):
    """The paper's simple rule: 'dose only when forced', i.e. only when the untreated tumour
    would otherwise exceed S_TOL.  Always feasible for our instance, so it is a good starting
    incumbent for branch-and-bound.  (It is a HEURISTIC: no guarantee of optimality in general.)"""
    X, prev = [], None
    for k in range(K):
        G = p["S0"] if k == 0 else p["A"] * prev
        x = 1 if G > p["S_TOL"] else 0          # forced to dose only if the tumour would be too big
        prev = G * (1 - p["RF"] * x)
        X.append(x)
    return X


def closed_form_bound(K, p=DEFAULT):
    """Fractional number of doses needed, from the formula y[K] = ln S0 + (K-1) ln A - lam*n <= ln S_TOL
    =>  n >= (ln S0 + (K-1) ln A - ln S_TOL) / lam.   Rounding UP gives a valid lower bound on the
    number of doses; it is also exactly the optimal value of the LP relaxation at the B&B root."""
    lam = -math.log(1 - p["RF"])
    return max(0.0, (math.log(p["S0"]) + (K - 1) * math.log(p["A"]) - math.log(p["S_TOL"])) / lam)


# =========================================================================================
# PART 2.  BRANCH-AND-BOUND
# =========================================================================================
#
# THE IDEA (read this first)
#   We want the best 0/1 vector X.  There are 2^K candidates - far too many to try one by one
#   for big K.  Branch-and-bound explores them as a SEARCH TREE but skips whole branches:
#
#       root:  no dose decided yet
#         |-- X[j] = 0 ...      (branch: fix one variable to 0)
#         |-- X[j] = 1 ...      (branch: fix the same variable to 1)
#       and so on, each node fixing one more X[j].
#
#   At each node we solve a RELAXATION: the same problem but the undecided X[j] may take any
#   value in [0, 1].  That is an ordinary LP (fast to solve) and gives three possible outcomes:
#
#     (a) the LP is INFEASIBLE         -> no schedule in this branch can work -> prune the branch.
#     (b) the LP's value is no better than the incumbent
#                                      -> nothing in this branch can beat what we have -> prune.
#     (c) the LP solution happens to be all 0/1
#                                      -> it is a real schedule: a new incumbent (best-so-far).
#     (d) otherwise (some X[j] fractional like 0.37)
#                                      -> pick the most fractional X[j] and BRANCH on it.
#
#   When no open nodes remain, the incumbent is PROVEN globally optimal.  The tree is explored
#   only where it could still matter, which is why B&B is usually far faster than brute force.
#
#   Two extra tricks used below:
#     * CEILING CUT: P*(number of doses) can only be a whole number of doses, so a node whose LP
#       bound is 20.06 can reach at best 21 doses.  We round the bound UP before comparing.
#     * WARM START : we start from the heuristic schedule as the incumbent, so pruning (b) works
#       from the very first node.
#
#   Node-selection rules (which open node to explore next):
#     "best"  : best-first - always take the open node with the smallest LP bound
#               (the most promising); needs a priority queue.
#     "depth" : depth-first - keep diving into the most recently created node; finds a full
#               schedule fast and uses little memory.
# -----------------------------------------------------------------------------------------
def _build_lp(K, p):
    """Build the pieces of the LP relaxation (they are the same at every node; only the BOUNDS
    on X change from node to node).  Variables, in order:  [X[1..K], y[1..K]]  (2K variables).

        equality rows (one per period k):   y[k] - y[k-1] + lam*X[k] = ln A     (k >= 2)
                                            y[1]          + lam*X[1] = ln S0
        objective:  minimise  X[1] + ... + X[K]   (number of doses; multiply by P for cost)
        bounds:     ln S_MIN <= y[k] <= ln S_TOL;     0 <= X[k] <= 1   (changed at each node)"""
    lam = -math.log(1 - p["RF"])
    A_eq = lil_matrix((K, 2 * K))      # coefficient matrix of the equality constraints
    b_eq = np.zeros(K)                 # right-hand sides
    for k in range(K):
        A_eq[k, K + k] = 1             # coefficient of y[k]
        A_eq[k, k] = lam               # coefficient of X[k]
        if k == 0:
            b_eq[k] = math.log(p["S0"])           # y[1] + lam X[1] = ln S0
        else:
            A_eq[k, K + k - 1] = -1               # - y[k-1]
            b_eq[k] = math.log(p["A"])            # ... = ln A
    c = np.r_[np.ones(K), np.zeros(K)]            # cost vector: 1 per dose, 0 on the y variables
    y_bounds = [(math.log(p["S_MIN"]), math.log(p["S_TOL"]))] * K   # size limits in log space
    return c, A_eq.tocsr(), b_eq, y_bounds


def branch_and_bound(K, p=DEFAULT, use_ceil=True, warm_start=True, select="best",
                     node_limit=200000, time_limit=60.0, trace=False):
    """Solve the scheduling problem by LP-based branch-and-bound.

    Options
      use_ceil   : apply the ceiling cut (round LP bounds up to a whole number of doses).
      warm_start : start with the 'dose only when forced' heuristic as the incumbent.
      select     : "best" (best-first) or "depth" (depth-first) node selection.
      node_limit / time_limit : safety stops; if hit, 'proved' is False (answer may be suboptimal).
      trace      : if True, record the CONVERGENCE HISTORY (one row per node: incumbent, global
                   lower bound, gap, ...) so we can plot how the search closes the gap.

    Returns a dict with the best schedule found, solver statistics and (optionally) the trace."""
    t0 = time.perf_counter()
    c, A_eq, b_eq, y_bounds = _build_lp(K, p)

    # ---- the incumbent (best feasible schedule so far); +infinity = "none found yet" ----
    best, best_X = math.inf, None
    if warm_start:
        h = heuristic(K, p)
        if feasible(h, p):
            best, best_X = sum(h), h            # the heuristic schedule is our first incumbent

    # ---- bookkeeping for analysis (all optional, none of it changes the search) ----
    nodes = 0                                    # number of nodes processed = number of LPs solved
    n_infeasible = n_bound = n_integral = n_branched = 0     # why nodes ended (see outcomes a-d)
    max_depth = 0                                # deepest level of the tree reached
    max_open = 0                                 # largest number of nodes waiting at once (memory use)
    root_bound = None                            # LP value at the root = the strongest simple lower bound
    history = [] if trace else None              # convergence record

    # ---- the list of OPEN nodes (created but not yet explored) ----
    # Each entry is (priority, counter, fixed, parent_bound):
    #   fixed        : dict {k: value} of the doses already decided on the path from the root
    #   parent_bound : LP value of the parent node = a valid lower bound for everything below it
    #   priority     : what we sort by.  Best-first uses the parent's bound (smallest first);
    #                  depth-first ignores it (it just takes the newest entry).
    #   counter      : unique tie-breaker so Python never has to compare the dicts.
    counter = 0
    open_nodes = [(0.0, counter, {}, 0.0)]       # start with the ROOT (nothing fixed; bound 0 is trivially valid)
    proved = True                                # becomes False if we stop early

    def global_lower_bound():
        """Smallest lower bound over everything still unexplored (only used for the trace).
        When nothing is left open, the incumbent itself is the bound => gap closes to zero."""
        if not open_nodes:
            return best
        m = min(e[3] for e in open_nodes)        # best (smallest) parent bound among open nodes
        if use_ceil:
            m = math.ceil(m - 1e-7)              # whole doses only
        return min(m, best)                      # a lower bound can never exceed the incumbent

    while open_nodes:
        # ---- stop early if we ran out of budget (the answer is then NOT proven optimal) ----
        if nodes >= node_limit or time.perf_counter() - t0 > time_limit:
            proved = False
            break

        # ---- pick the next node to explore ----
        # best-first: heappop returns the entry with the smallest priority (most promising bound)
        # depth-first: pop() returns the most recently appended entry (dive deeper)
        entry = heapq.heappop(open_nodes) if select == "best" else open_nodes.pop()
        fixed = entry[2]
        nodes += 1
        max_depth = max(max_depth, len(fixed))

        # ---- build the variable bounds for THIS node ----
        # decided doses get lower = upper = their value; undecided doses stay free in [0, 1]
        bnds = [(fixed[k], fixed[k]) if k in fixed else (0, 1) for k in range(K)] + y_bounds

        # ---- solve the LP relaxation of this node ----
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")

        event = None                              # what happened at this node (for the trace)
        if res.status != 0:
            # (a) no solution even with fractional doses -> nothing below this node can be feasible
            n_infeasible += 1
            event = "infeasible"
        else:
            bound = res.fun                       # best possible number of doses in this branch (fractional)
            if root_bound is None:
                root_bound = bound                # the first LP solved is the root relaxation
            # Doses are whole numbers, so a branch with bound 20.06 cannot do better than 21.
            lb = math.ceil(bound - 1e-7) if use_ceil else bound
            if lb >= best - 1e-9:
                # (b) this branch cannot beat the incumbent -> discard it ("prune by bound")
                n_bound += 1
                event = "pruned_by_bound"
            else:
                x = res.x[:K]                     # the LP's dose values (may be fractional)
                frac = np.abs(x - np.rint(x))     # distance of each dose from the nearest whole number
                j = int(np.argmax(frac))          # index of the MOST fractional dose
                if frac[j] < 1e-6:
                    # (c) every dose is 0 or 1 (up to rounding): a genuine schedule
                    n_integral += 1
                    event = "integral"
                    X = [int(round(v)) for v in x]
                    # double-check in the ORIGINAL model, then keep it if it beats the incumbent
                    if feasible(X, p) and sum(X) < best:
                        best, best_X = sum(X), X
                else:
                    # (d) branch on dose j: create two children, one with X[j]=0 and one with X[j]=1
                    n_branched += 1
                    event = "branched"
                    for val in (0, 1):
                        child = dict(fixed)       # copy the parent's decisions ...
                        child[j] = val            # ... and add the new one
                        counter += 1
                        entry_new = (bound if select == "best" else 0.0, counter, child, bound)
                        if select == "best":
                            heapq.heappush(open_nodes, entry_new)   # keep the queue sorted by bound
                        else:
                            open_nodes.append(entry_new)            # depth-first: newest last => popped first
        max_open = max(max_open, len(open_nodes))

        # ---- convergence record: one row per node ----
        if trace:
            lbg = global_lower_bound()
            history.append(dict(node=nodes, time=time.perf_counter() - t0, depth=len(fixed),
                                event=event,
                                lp_bound=None if res.status != 0 else float(res.fun),
                                incumbent=None if best == math.inf else int(best),
                                lower_bound=None if lbg == math.inf else float(lbg),
                                open_nodes=len(open_nodes)))

    return dict(doses=None if best_X is None else int(best), X=best_X, nodes=nodes,
                lp_solves=nodes, time=time.perf_counter() - t0, proved=proved,
                root_bound=root_bound, n_infeasible=n_infeasible, n_pruned_bound=n_bound,
                n_integral=n_integral, n_branched=n_branched, max_depth=max_depth,
                max_open=max_open, history=history)


# =========================================================================================
# PART 3.  DYNAMIC PROGRAMMING
# =========================================================================================
#
# THE IDEA (read this first)
#   Dynamic programming solves a problem made of STAGES by building the answer stage by stage,
#   reusing partial answers instead of recomputing them.  It works when the problem has
#   "optimal substructure": what you should do from now on depends only on a small summary of
#   the past (the STATE), not on the whole history.
#
#   Our problem: stages are the periods k = 1..K.  From the log formula
#       y[k] = ln S0 + (k-1) ln A - lam * n   (n = doses given so far)
#   the tumour size after period k is fixed once we know (k, n).  So the STATE is the pair
#   (k, n) - we do not need to remember WHICH weeks the doses were given.
#
#   Define the table      reach[k][n] = True  if  some allowed schedule for periods 1..k uses
#                                               exactly n doses and respects the size limits.
#   RECURSION (how one stage is built from the previous one):
#       to be at (k, n) we either
#           gave NO dose in period k  -> we were at (k-1, n)     (n unchanged), or
#           gave a dose in period k   -> we were at (k-1, n-1)   (one more dose),
#       and the new state (k, n) must itself satisfy ln S_MIN <= y(k, n) <= ln S_TOL.
#       so   reach[k][n] = allowed(k, n)  AND  ( reach[k-1][n]  OR  reach[k-1][n-1] ).
#   ANSWER: the smallest n with reach[K][n] = True.
#   BACKTRACKING: walk backwards from (K, n_best) to recover WHICH weeks had a dose.
#
#   Work needed: the table has K x (K+1) entries and each is a constant-time check, so the
#   cost grows like K^2 - tiny compared with the 2^K schedules brute force would examine.
#   Unlike branch-and-bound, DP does not search a tree and never needs an LP: it just fills
#   in a table.  Because every state is considered, the result is a GLOBAL optimum.
# -----------------------------------------------------------------------------------------
def dp_solve(K, p=DEFAULT, keep_table=False):
    """Exact dynamic program (see the explanation above).
    keep_table=True also returns the whole reach[k][n] table, for analysis and plotting."""
    lam = -math.log(1 - p["RF"])
    lnA = math.log(p["A"])
    lnS0 = math.log(p["S0"])
    # allowed range of y; the 1e-12 tolerates floating-point rounding exactly on a limit
    zL, zU = math.log(p["S_MIN"]) - 1e-12, math.log(p["S_TOL"]) + 1e-12
    t0 = time.perf_counter()

    n_idx = np.arange(K + 1)                       # all possible dose counts 0..K (as an array)

    def allowed(k):
        """For period index k (0-based, so k = 0 is week 1): a boolean array over n telling
        whether the state (k, n) respects the size limits.  Uses y(k, n) = ln S0 + k ln A - lam n."""
        y = lnS0 + k * lnA - lam * n_idx
        return (y >= zL) & (y <= zU)

    reach = np.zeros((K, K + 1), dtype=bool)       # reach[k][n]: see explanation above
    # ---- first period: either no dose (n = 0) or one dose (n = 1) ----
    ok0 = allowed(0)
    reach[0, 0] = ok0[0]
    reach[0, 1] = ok0[1]
    # ---- later periods: apply the recursion to ALL n at once with array operations ----
    for k in range(1, K):
        r = reach[k - 1]                           # which counts were reachable one period ago
        nxt = r.copy()                             # "no dose in period k": n stays the same
        nxt[1:] |= r[:-1]                          # "dose in period k":    n-1 becomes n   (shift by one)
        reach[k] = nxt & allowed(k)                # keep only the states that respect the size limits

    # ---- read off the answer: the smallest dose count reachable at the last period ----
    good = np.where(reach[K - 1])[0]
    if good.size == 0:                             # no n works -> the instance is INFEASIBLE
        return dict(doses=None, X=None, time=time.perf_counter() - t0,
                    states=int(reach.sum()), table=reach if keep_table else None)
    n = int(good[0])

    # ---- backtracking: recover one optimal schedule ----
    X = [0] * K
    m = n                                          # doses still to place, going backwards in time
    for k in range(K - 1, 0, -1):
        # If the state (k-1, m-1) is reachable we may place a dose in period k.  Preferring this
        # pushes doses as LATE as possible, which matches the paper's "dose only when forced".
        if m >= 1 and reach[k - 1, m - 1]:
            X[k] = 1
            m -= 1
    X[0] = m                                       # whatever is left (0 or 1 dose) belongs to period 1
    return dict(doses=n, X=X, time=time.perf_counter() - t0,
                states=int(reach.sum()),           # how many (k, n) states were reachable
                table=reach if keep_table else None)


def dp_count_optimal(K, p=DEFAULT):
    """Count HOW MANY different schedules attain the optimum (global optima are not unique).
    Same table as dp_solve, but instead of True/False we count the number of ways to reach each state:
        ways[k][n] = allowed(k, n) * ( ways[k-1][n] + ways[k-1][n-1] ).
    Python integers (dtype=object) are used so large counts never overflow."""
    lam = -math.log(1 - p["RF"]); lnA = math.log(p["A"]); lnS0 = math.log(p["S0"])
    zL, zU = math.log(p["S_MIN"]) - 1e-12, math.log(p["S_TOL"]) + 1e-12
    n_idx = np.arange(K + 1)
    allowed = lambda k: (lnS0 + k * lnA - lam * n_idx >= zL) & (lnS0 + k * lnA - lam * n_idx <= zU)
    ways = np.zeros(K + 1, dtype=object)
    ok0 = allowed(0)
    ways[0] = 1 if ok0[0] else 0                   # one way: no dose in period 1
    ways[1] = 1 if ok0[1] else 0                   # one way: dose in period 1
    for k in range(1, K):
        new = ways.copy()                          # ways to arrive with no dose in period k
        new[1:] = new[1:] + ways[:-1]              # plus ways to arrive with a dose in period k
        ok = allowed(k)
        ways = np.array([w if o else 0 for w, o in zip(new, ok)], dtype=object)
    for n in range(K + 1):                         # the smallest n that has at least one way is the optimum
        if ways[n] > 0:
            return n, int(ways[n])
    return None, 0
