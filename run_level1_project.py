"""
===========================================================================================
 run_level1_project.py - the main experiment driver
===========================================================================================
Runs four experiments and writes their results to results/level1/ :
  1. VALIDATION   : every solver is compared with brute force for K = 2..18 (a correctness check).
  2. SCALABILITY  : how solve time and B&B node count grow with the horizon K (the course's
                    "scalability analysis"), for our DP, three B&B variants and two HiGHS MILPs.
  3. PARETO       : the trade-off between number of doses and the tumour-size limit S_TOL, found
                    by the "epsilon-constraint" method (fix one objective as a constraint, optimise
                    the other, repeat for many values).
  4. SCHEDULES    : the headline optimal schedules for K = 8 (hand-checkable) and K = 52 (paper).

Usage:  python3 run_level1_project.py      (takes a few minutes; variant A hits its time limit)
Output: validation_bruteforce.csv, scalability_level1.csv, pareto_doses_vs_tumour_cap.csv, schedules.json

B&B VARIANTS (see bnb_dp.py for the algorithm)
  A  best-first, NO ceiling cut, cold start  : the plain textbook method  (slow)
  B  depth-first, ceiling cut, cold start    : the cut alone helps a lot
  C  best-first, ceiling cut, warm start     : heuristic incumbent + cut -> proof at the root node
===========================================================================================
"""
import json, os, statistics, sys, time, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bnb_dp import DEFAULT, branch_and_bound, dp_solve, heuristic, feasible, simulate
import level1_exponential as paper      # paper-style big-M MILP + log MILP (HiGHS) from the Level 1 script

OUT = os.path.join(HERE, "results", "level1"); os.makedirs(OUT, exist_ok=True)


def med(fn, reps=15):
    """Median wall-clock time of fn() over `reps` runs.  Very fast functions (the DP takes microseconds)
    are noisy, so we repeat them and take the median, which ignores the occasional slow outlier."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


# ---------------------------------------------------------------- 1. validation vs brute force
def brute(K):
    """Exhaustive search over all 2^K schedules: slow but obviously correct -> ground truth."""
    best = None
    for X in itertools.product([0, 1], repeat=K):
        if (best is None or sum(X) < best) and feasible(X):
            best = sum(X)
    return best

val = []
for K in range(2, 19):
    row = dict(K=K, brute=brute(K), dp=dp_solve(K)["doses"],
               bnb_warm=branch_and_bound(K)["doses"],
               bnb_cold=branch_and_bound(K, warm_start=False, select="depth")["doses"],
               heuristic=sum(heuristic(K)))
    val.append(row)
val = pd.DataFrame(val); val.to_csv(os.path.join(OUT, "validation_bruteforce.csv"), index=False)
# If every method finds the same number of doses, each row has exactly 1 distinct value.
assert (val.drop(columns="K").nunique(axis=1) == 1).all(), "solvers disagree"
print("validation: all solvers agree with brute force for K = 2..18")

# ---------------------------------------------------------------- 2. scalability + ablation
# 'Ablation' = switch features of B&B on/off one at a time to see what each contributes.
Ks = [4, 8, 12, 16, 20, 24, 28, 32, 40, 52, 75, 100, 150, 200, 300]
rows = []
for K in Ks:
    r = dict(K=K)
    d = dp_solve(K); r["doses"] = d["doses"]; r["dp_time"] = med(lambda: dp_solve(K))
    for name, kw in [("A_best_nocut_cold", dict(use_ceil=False, warm_start=False, select="best", time_limit=20)),
                     ("B_depth_ceil_cold", dict(use_ceil=True, warm_start=False, select="depth", time_limit=60)),
                     ("C_best_ceil_warm", dict(use_ceil=True, warm_start=True, select="best", time_limit=60))]:
        if name.startswith("A") and K > 52:          # variant A already hits its 20 s limit from K = 16
            continue
        res = branch_and_bound(K, **kw)
        r[name + "_time"] = res["time"]; r["%s_nodes" % name] = res["nodes"]
        r[name + "_proved"] = res["proved"]; r[name + "_doses"] = res["doses"]
    if K <= 28:                                       # big-M model: hopeless beyond ~K=28 (20 s limit)
        res, t = paper.milp_bigM(K, time_limit=20)
        r["bigM_time"] = t; r["bigM_proved"] = (res.status == 0); r["bigM_doses"] = paper.doses(res, K)
    res, t = paper.milp_log(K, time_limit=60)
    r["logMILP_time"] = t; r["logMILP_doses"] = paper.doses(res, K)
    rows.append(r); print("K=%d done" % K, flush=True)
sc = pd.DataFrame(rows); sc.to_csv(os.path.join(OUT, "scalability_level1.csv"), index=False)
# consistency: every solver that finished must report the DP optimum (the DP is exact)
for col in [c for c in sc.columns if c.endswith("_doses") and c != "doses"]:
    ok = sc[col].isna() | (sc[col] == sc["doses"])
    print("agree with DP:", col, bool(ok.all()))
print(sc.round(4).to_string())

# ---------------------------------------------------------------- 3. epsilon-constraint Pareto: doses vs max tumour size
# Two competing goals: few doses (cost, side effects) vs a small maximum tumour.  For each cap S_TOL we
# find the minimum number of doses -> one point of the Pareto front.  Geometric spacing covers many scales.
K = 52
par = []
for s_tol in np.unique(np.round(np.geomspace(40, 4000, 90))):
    p = dict(DEFAULT, S_TOL=float(s_tol))              # same parameters but a different cap
    d = dp_solve(K, p); b = branch_and_bound(K, p)
    if d["X"] is not None: assert feasible(d["X"], p)   # re-check in the original nonlinear model
    par.append(dict(S_TOL=float(s_tol), doses=d["doses"], cost=None if d["doses"] is None else DEFAULT["P"] * d["doses"],
                    bnb_doses=b["doses"], max_S=None if d["X"] is None else float(simulate(d["X"], p).max())))
par = pd.DataFrame(par); par.to_csv(os.path.join(OUT, "pareto_doses_vs_tumour_cap.csv"), index=False)
print(par.to_string())

# ---------------------------------------------------------------- 4. headline schedule K = 52 and hand example K = 8
sched = {}
for K in (8, 52):
    d = dp_solve(K); X = d["X"]
    sched[K] = dict(doses=d["doses"], cost=DEFAULT["P"] * d["doses"], weeks_with_dose=[i + 1 for i, x in enumerate(X) if x],
                    S_after_dosing=[round(float(v), 2) for v in simulate(X)],
                    S_untreated=[round(float(v), 2) for v in simulate([0] * K)])
json.dump(sched, open(os.path.join(OUT, "schedules.json"), "w"), indent=1)
print("K=8:", sched[8]["weeks_with_dose"], " K=52:", sched[52]["weeks_with_dose"])
