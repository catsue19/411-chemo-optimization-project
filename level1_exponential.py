"""
===========================================================================================
 level1_exponential.py - the problem as a MILP solved by a GENERAL-PURPOSE solver (HiGHS)
===========================================================================================
LEVEL 1 - Chemotherapy dose scheduling, exponential tumour growth, binary doses.
(Model from Braeutigam 2024, Comput. Ind. Eng. 191:110078, eqs. 1-5, Section 3.)

WHAT THIS FILE IS FOR
  bnb_dp.py contains OUR OWN algorithms (branch-and-bound and dynamic programming).
  This file contains the "off-the-shelf" alternative: write the problem as a Mixed-Integer
  Linear Program (MILP) and hand it to a professional solver (HiGHS, via scipy.optimize.milp).
  It is used in the project as a COMPARISON: how does a commercial-grade solver behave on
  (a) the paper's linearisation and (b) our log-transformed model?

THE PROBLEM
  Decision:  X[k] = 1 if a dose is given at the start of week k, else 0      (k = 1..K)
  Dynamics:  G[k] = S0 (k=1),  G[k] = A * S[k-1] (k>1)     tumour size BEFORE dosing ("grown")
             S[k] = G[k] * (1 - RF * X[k])                  tumour size AFTER dosing
  Minimise:  P * sum_k X[k]                                 total drug cost
  Subject to S_MIN <= S[k] <= S_TOL  for all k              tumour may be neither too big nor
                                                            over-shrunk (quality-of-life limit)

WHY IT IS HARD AS WRITTEN
  S[k] = G[k] * (1 - RF * X[k]) multiplies two unknowns (G and X).  A product of two unknown
  variables is NONLINEAR, and X is integer -> the model is a MINLP (mixed-integer nonlinear
  program).  Solvers for MILPs are far more mature than solvers for MINLPs, so the usual trick
  is to rewrite the model to be LINEAR.  Two ways are compared below:
    A) the paper's "big-M" linearisation of the product G*X          (weak, slow)
    B) a log transform  y = ln S  that makes the dynamics exactly linear   (our technique)

GLOSSARY
  MILP     : linear objective, linear constraints, some variables must be integers.
  LP relaxation : the MILP with the integer requirement dropped (X may be 0.3, say).  It gives a
                  bound on the true answer.  A TIGHT relaxation (bound close to the true optimum)
                  means little searching is needed; a WEAK one means a lot of searching.
  big-M    : a trick to switch a constraint "on/off" with a binary variable and a large constant M.
             The larger M is, the weaker the LP relaxation.
Run:  python3 level1_exponential.py
===========================================================================================
"""
import itertools, math, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

# ---- parameters (paper, Section 3.1; fictitious drug) ----
# S0    : initial tumour size            A     : weekly growth factor (1.5 = +50 % per week)
# RF    : fraction killed by a dose      S_TOL : largest tumour size allowed
# S_MIN : smallest tumour size allowed   P     : cost of one dose
S0, A, RF, S_TOL, S_MIN, P = 50.0, 1.5, 0.6, 500.0, 10.0, 10.0


# =====================================================================  simulation / checks
def simulate(X):
    """Return tumour size after dosing S[k] for a 0/1 schedule X (original nonlinear model).
    This is the 'ground truth' model: every schedule found by any method is re-checked with it."""
    S = []
    for k, x in enumerate(X):
        G = S0 if k == 0 else A * S[-1]      # grow from the previous week's size (week 1 starts at S0)
        S.append(G * (1 - RF * x))           # a dose removes the fraction RF of the tumour
    return np.array(S)


def feasible(X):
    """True if the tumour stays inside [S_MIN, S_TOL] in every week (1e-9 = rounding tolerance)."""
    S = simulate(X)
    return S.max() <= S_TOL + 1e-9 and S.min() >= S_MIN - 1e-9


def heuristic(K):
    """Paper's heuristic: give a dose only when the tumour would otherwise exceed S_TOL.
    It is a quick, simple rule ('procrastinate until forced'); it gives a FEASIBLE schedule but
    nothing guarantees it is the cheapest one."""
    X, S_prev = [], None
    for k in range(K):
        G = S0 if k == 0 else A * S_prev
        x = 1 if G > S_TOL else 0            # forced dose: without it the size would exceed S_TOL
        S_prev = G * (1 - RF * x)
        X.append(x)
    return X


def brute_force(K):
    """Try all 2^K schedules (only for small K) -> true optimum, used to validate the MILPs.
    itertools.product([0,1], repeat=K) lists every 0/1 string of length K.
    The 'sum(X) < best' test is checked first because it is cheaper than the feasibility check."""
    best = None
    for X in itertools.product([0, 1], repeat=K):
        if (best is None or sum(X) < best) and feasible(X):
            best = sum(X)
    return best


# =====================================================================  MILP A: paper's big-M
def milp_bigM(K, time_limit=20):
    """The paper's linearisation (its eqs 6-9).

    IDEA.  Introduce R[k] = RF * X[k] * G[k]  (the amount of tumour removed by the dose).
    Then  S[k] = G[k] - R[k]  is LINEAR.  We must force R to equal the product, using only
    linear inequalities.  Because X is 0 or 1 there are only two cases:
        X = 0  ->  R must be 0
        X = 1  ->  R must be RF * G
    The four inequalities below encode exactly that, using the known bounds
    glo <= G <= ghi as the 'big-M' constants:
        (6)  RF*glo*X <= R           (7)  R <= RF*ghi*X                 [R is squeezed to 0 when X = 0]
        (8),(9)  G - R/RF  stays between glo*(1-X) and ghi*(1-X)        [R = RF*G when X = 1]
    The weakness: with X relaxed to a fraction, these constraints allow many 'fake' combinations,
    so the LP relaxation is loose and the solver must branch a lot.

    Variables, in one long vector:  [ X(K) | G(K) | R(K) | S(K) ]   (4K variables in total)."""
    n = 4 * K
    iX, iG, iR, iS = 0, K, 2 * K, 3 * K          # starting index of each block in the vector
    rows, lb, ub = [], [], []                     # constraint k:   lb <= (row . variables) <= ub

    def add(coefs, lo, hi):
        """Store one constraint. coefs = {variable index: coefficient}; lo/hi = bounds on the sum."""
        rows.append(coefs); lb.append(lo); ub.append(hi)

    for k in range(K):
        glo = S0 if k == 0 else A * S_MIN          # smallest possible G[k] -> big-M constant
        ghi = S0 if k == 0 else A * S_TOL          # largest possible G[k]  -> big-M constant
        if k == 0:
            add({iG + k: 1}, S0, S0)                                    # G1 = S0
        else:
            add({iG + k: 1, iS + k - 1: -A}, 0, 0)                      # G_k = A S_{k-1}
        add({iS + k: 1, iG + k: -1, iR + k: 1}, 0, 0)                   # S = G - R
        add({iX + k: RF * glo, iR + k: -1}, -np.inf, 0)                 # (6)
        add({iR + k: 1, iX + k: -RF * ghi}, -np.inf, 0)                 # (7)
        add({iG + k: 1, iR + k: -1 / RF, iX + k: glo}, glo, np.inf)     # (8)
        add({iG + k: 1, iR + k: -1 / RF, iX + k: ghi}, -np.inf, ghi)    # (9)
    # Convert the list of constraints to a sparse matrix (most entries are zero, so sparse is efficient).
    M = lil_matrix((len(rows), n))
    for r, coefs in enumerate(rows):
        for j, a in coefs.items():
            M[r, j] = a
    c = np.r_[np.full(K, P), np.zeros(3 * K)]            # cost vector: P for each X, 0 for the rest
    integrality = np.r_[np.ones(K), np.zeros(3 * K)]     # 1 = integer variable (only the X's)
    bounds = Bounds(np.r_[np.zeros(3 * K), np.full(K, S_MIN)],               # lower bounds
                    np.r_[np.ones(K), np.full(2 * K, 2000.0), np.full(K, S_TOL)])  # upper bounds
    t0 = time.perf_counter()
    res = milp(c, constraints=LinearConstraint(M.tocsr(), lb, ub), integrality=integrality,
               bounds=bounds, options={"time_limit": time_limit})
    return res, time.perf_counter() - t0


# =====================================================================  MILP B: log transform
def milp_log(K, time_limit=60):
    """OUR reformulation.  Take logarithms: y[k] = ln S[k].
    Growth multiplies by A, so ln adds ln A.  A dose multiplies by (1-RF), so ln adds ln(1-RF) = -lam.
    Therefore
          y[k] = y[k-1] + ln A - lam * X[k]        with  lam = -ln(1-RF) = 0.916
    is EXACTLY LINEAR - no products, no big-M, nothing approximate.  The size limits become simple
    bounds  ln S_MIN <= y[k] <= ln S_TOL.
    Variables [X(K) | y(K)] (2K variables).  Its LP relaxation is tight: it equals
    (ln S0 + (K-1) ln A - ln S_TOL)/lam  doses, and rounding that UP gives the optimum."""
    lam = -math.log(1 - RF)
    M = lil_matrix((K, 2 * K)); lb = np.zeros(K); ub = np.zeros(K)
    for k in range(K):
        M[k, K + k] = 1; M[k, k] = lam                    # y_k + lam X_k = ...
        if k == 0:
            lb[k] = ub[k] = math.log(S0)                   # y_1 + lam X_1 = ln S0
        else:
            M[k, K + k - 1] = -1
            lb[k] = ub[k] = math.log(A)                    # y_k - y_{k-1} + lam X_k = ln A
    c = np.r_[np.full(K, P), np.zeros(K)]
    integrality = np.r_[np.ones(K), np.zeros(K)]
    bounds = Bounds(np.r_[np.zeros(K), np.full(K, math.log(S_MIN))],
                    np.r_[np.ones(K), np.full(K, math.log(S_TOL))])
    t0 = time.perf_counter()
    res = milp(c, constraints=LinearConstraint(M.tocsr(), lb, ub), integrality=integrality,
               bounds=bounds, options={"time_limit": time_limit})
    return res, time.perf_counter() - t0


def doses(res, K):
    """Number of doses in a solver result (None if the solver found nothing). round() removes 0.9999999 noise."""
    return None if res.x is None else int(round(res.x[:K].sum()))


# =====================================================================  demo
if __name__ == "__main__":
    # ---- 1. hand-checkable example, K = 8 ----
    K = 8
    res, _ = milp_log(K)
    X = np.rint(res.x[:K]).astype(int)
    print("K = 8 example: optimal schedule X =", X.tolist(), "-> doses =", X.sum(), ", cost =", P * X.sum())
    S_none = simulate([0] * K)
    print("  S(k) with no drug :", np.round(S_none, 1), "<- exceeds", S_TOL, "at k =",
          int(np.argmax(S_none > S_TOL)) + 1)
    print("  S(k) with schedule:", np.round(simulate(X), 1), "feasible:", feasible(X))

    # ---- 2. validation + scalability ----
    # 'closed' is the formula-based answer: ceiling of the LP bound.  It should equal every solver's answer.
    print("\n  K | brute  heuristic | log-MILP doses  time(s) | big-M MILP doses  time(s)  proved? | ln-formula")
    for K in [4, 8, 12, 16, 20, 24, 28, 32, 40, 52]:
        bf = brute_force(K) if K <= 16 else None         # brute force explodes beyond K ~ 20
        h = sum(heuristic(K))
        rl, tl = milp_log(K)
        closed = max(0, math.ceil((math.log(S0) + (K - 1) * math.log(A) - math.log(S_TOL)) / -math.log(1 - RF) - 1e-12))
        if K <= 32:
            rb, tb = milp_bigM(K)
            big = "%4s %9.2f %9s" % (doses(rb, K), tb, rb.status == 0)   # status 0 = proved optimal
        else:
            big = "   (skipped)"                          # too slow to be worth waiting for
        print("%3d | %5s %8d | %10d %10.3f | %s | %d" % (K, bf if bf is not None else "-", h,
                                                       doses(rl, K), tl, big, closed))
