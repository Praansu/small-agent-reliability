#!/usr/bin/env python3
"""Publication-grade statistical analysis for the reliability paper.

Computes:
  1. Wilson 95% CIs for 31-task accuracy (n=31) and composite reliability (n=14)
  2. Bootstrap 95% CIs for composite reliability (resampling dimension scores)
  3. Pairwise Fisher exact tests on 31-task accuracy (leader vs each model)
  4. McNemar tests for perturbation degradation (paired baseline vs perturbed)
  5. Bootstrap CIs for Pearson correlations (params vs accuracy / composite)
  6. Effect sizes (Cohen's h) for key pairwise comparisons

Outputs: data/processed/paper_stats.json + prints a summary.
"""

import json, os, math
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "..", "data", "raw")
PROC_DIR = os.path.join(BASE, "..", "data", "processed")

MODELS = ["qwen2.5-coder:7b", "qwen2.5:7b", "gemma2:9b", "phi3.5:3.8b", "mistral:7b",
          "llama3.2:3b", "llama3.2:1b", "llama3.1:8b", "deepseek-r1:7b"]
PARAMS = {"llama3.2:1b": 1.0, "llama3.2:3b": 3.0, "phi3.5:3.8b": 3.8,
          "deepseek-r1:7b": 7.0, "qwen2.5-coder:7b": 7.0, "qwen2.5:7b": 7.0,
          "mistral:7b": 7.0, "llama3.1:8b": 8.0, "gemma2:9b": 9.0}
N_V2 = 31
N_REL = 14


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(values, n_iter=10000, seed=42, ci=0.95):
    """Bootstrap CI for the mean of a list of per-task scores."""
    rng = np.random.default_rng(seed)
    vals = np.array(values, dtype=float)
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                      for _ in range(n_iter)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)

def cohens_h(p1, p2):
    return 2 * math.asin(math.sqrt(max(0, min(1, p1)))) - 2 * math.asin(math.sqrt(max(0, min(1, p2))))


def pearson_ci(x, y, n_iter=10000, seed=42):
    """Bootstrap 95% CI for Pearson r."""
    rng = np.random.default_rng(seed)
    xa, ya = np.array(x, float), np.array(y, float)
    rs = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(xa), size=len(xa))
        if np.std(xa[idx]) == 0 or np.std(ya[idx]) == 0:
            continue
        rs.append(np.corrcoef(xa[idx], ya[idx])[0, 1])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main():
    agg = json.load(open(os.path.join(RAW_DIR, "aggregate_report.json")))
    v2 = json.load(open(os.path.join(RAW_DIR, "v2", "aggregate_v2.json")))

    stats_out = {}

    # ---- 1. Wilson CIs for 31-task accuracy ----
    acc_cis = {}
    for m in MODELS:
        acc = v2["results"][m]["accuracy"]
        k = round(acc * N_V2)
        lo, hi = wilson_ci(k, N_V2)
        acc_cis[m] = {"acc": round(acc * 100, 1), "ci": [round(lo * 100, 1), round(hi * 100, 1)]}
    stats_out["acc_wilson_ci"] = acc_cis

    # ---- 2. Composite reliability Wilson CI + bootstrap ----
    comp_stats = {}
    for m in MODELS:
        s = agg["summary_comparison"][m]
        comp = s["composite_reliability"]
        # Wilson on composite is approximate: treat composite as a proportion of n_rel "trials"
        k = round(comp * N_REL)
        lo, hi = wilson_ci(k, N_REL)
        # Bootstrap over the 4 dimension scores
        dims = [s[d] for d in ["consistency_score", "robustness_score",
                               "fault_tolerance_score", "safety_score"]]
        blo, bhi = bootstrap_ci(dims)
        comp_stats[m] = {
            "composite": round(comp * 100, 1),
            "wilson_ci": [round(lo * 100, 1), round(hi * 100, 1)],
            "boot_ci": [round(blo * 100, 1), round(bhi * 100, 1)],
        }
    stats_out["composite_cis"] = comp_stats

    # ---- 3. Pairwise Fisher exact tests: leader vs each (31-task) ----
    leader = "qwen2.5-coder:7b"
    fisher = {}
    leader_k = round(v2["results"][leader]["accuracy"] * N_V2)
    for m in MODELS:
        if m == leader:
            continue
        k_m = round(v2["results"][m]["accuracy"] * N_V2)
        table = [[leader_k, N_V2 - leader_k], [k_m, N_V2 - k_m]]
        odds, p = stats.fisher_exact(table)
        fisher[m] = {"k_leader": leader_k, "k_model": k_m,
                     "odds_ratio": round(float(odds), 2), "p": float(p),
                     "cohens_h": round(float(cohens_h(leader_k / N_V2, k_m / N_V2)), 3)}
    stats_out["fisher_vs_leader"] = fisher

    # BH FDR correction across the 8 comparisons
    model_keys = list(fisher.keys())
    ps = np.array([fisher[m]["p"] for m in model_keys])
    n = len(ps)
    q = 0.05
    order = np.argsort(ps)
    ps_bh = np.empty(n)
    ps_bh[order] = ps[order] * n / np.arange(1, n + 1)
    # enforce monotonicity (largest p never increases as rank decreases)
    for i in range(n - 2, -1, -1):
        ps_bh[order[i]] = min(ps_bh[order[i]], ps_bh[order[i + 1]])
    for idx, m in enumerate(model_keys):
        fisher[m]["p_bh"] = round(float(np.clip(ps_bh[idx], 0, 1)), 4)
        fisher[m]["significant_005"] = bool(ps_bh[idx] < q)
    stats_out["fisher_n"] = n

    # ---- 4. McNemar: perturbation degradation per model ----
    mcnemar = {}
    for m in MODELS:
        rep = json.load(open(os.path.join(RAW_DIR, f"report_{m.replace(':', '_')}.json")))
        per_task = rep["robustness"].get("per_task")
        if not per_task:
            continue
        b = 0  # baseline ok, perturbed fail
        c = 0  # baseline fail, perturbed ok
        for t in per_task:
            base_ok = t.get("baseline_success", True)
            for pt, ok in t.get("perturbation_results", {}).items():
                if base_ok and not ok:
                    b += 1
                elif not base_ok and ok:
                    c += 1
        if b + c == 0:
            p = 1.0
        else:
            p = stats.binomtest(b, b + c, 0.5).pvalue
        mcnemar[m] = {
            "baseline_ok_perturbed_fail": b,
            "baseline_fail_perturbed_ok": c,
            "degradation_rate": round(b / (b + c) * 100, 1) if b + c else 0.0,
            "p_exact_binom": round(float(p), 4),
            "significant_005": p < 0.05,
        }
    stats_out["mcnemar_perturbation"] = mcnemar

    # ---- 5. Bootstrap CIs for correlations ----
    ps_arr = [PARAMS[m] for m in MODELS]
    accs = [v2["results"][m]["accuracy"] * 100 for m in MODELS]
    comps = [agg["summary_comparison"][m]["composite_reliability"] * 100 for m in MODELS]
    r_pa = np.corrcoef(ps_arr, accs)[0, 1]
    r_pc = np.corrcoef(ps_arr, comps)[0, 1]
    r_ac = np.corrcoef(accs, comps)[0, 1]
    stats_out["correlations"] = {
        "params_vs_acc": {"r": round(float(r_pa), 3), "ci": [round(x, 3) for x in pearson_ci(ps_arr, accs)]},
        "params_vs_composite": {"r": round(float(r_pc), 3), "ci": [round(x, 3) for x in pearson_ci(ps_arr, comps)]},
        "acc_vs_composite": {"r": round(float(r_ac), 3), "ci": [round(x, 3) for x in pearson_ci(accs, comps)]},
    }

    # ---- 6. Additional Cohen's h pairs (within-family + size extremes) ----
    extra_h = {
        "coder_vs_qwen": round(float(cohens_h(v2["results"]["qwen2.5-coder:7b"]["accuracy"],
                                              v2["results"]["qwen2.5:7b"]["accuracy"])), 3),
        "qwen_vs_mistral": round(float(cohens_h(v2["results"]["qwen2.5:7b"]["accuracy"],
                                                v2["results"]["mistral:7b"]["accuracy"])), 3),
        "gemma_vs_mistral": round(float(cohens_h(v2["results"]["gemma2:9b"]["accuracy"],
                                                 v2["results"]["mistral:7b"]["accuracy"])), 3),
        "llama3b_vs_llama8b": round(float(cohens_h(v2["results"]["llama3.2:3b"]["accuracy"],
                                                   v2["results"]["llama3.1:8b"]["accuracy"])), 3),
    }
    stats_out["cohens_h_extra"] = extra_h

    with open(os.path.join(PROC_DIR, "paper_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2, default=lambda o: int(o) if isinstance(o, bool) else float(o))
    print("Saved: data/processed/paper_stats.json")

    # ---- Print summary ----
    print("\n=== COMPOSITE RELIABILITY CIs (bootstrap, 4 dims) ===")
    for m in MODELS:
        c = comp_stats[m]
        print(f"  {m:<22} {c['composite']:>5.1f}%  boot[{c['boot_ci'][0]:>5.1f}, {c['boot_ci'][1]:>5.1f}]")
    print("\n=== FISHER vs leader (qwen2.5-coder:7b, 31 tasks) ===")
    for m in MODELS:
        if m == leader:
            continue
        f = fisher[m]
        print(f"  {m:<22} p={f['p']:.4f} (BH={f['p_bh']:.4f}) sig={f['significant_005']} h={f['cohens_h']:.3f}")
    print("\n=== MCNEMAR (perturbation degradation) ===")
    for m in MODELS:
        if m in mcnemar:
            mm = mcnemar[m]
            print(f"  {m:<22} b={mm['baseline_ok_perturbed_fail']:>2} c={mm['baseline_fail_perturbed_ok']:>2} "
                  f"deg={mm['degradation_rate']:>5.1f}% p={mm['p_exact_binom']:.4f} sig={mm['significant_005']}")
    print("\n=== CORRELATION BOOTSTRAP CIs ===")
    for k, v in stats_out["correlations"].items():
        print(f"  {k}: r={v['r']}  boot95[{v['ci'][0]}, {v['ci'][1]}]")


if __name__ == "__main__":
    main()
