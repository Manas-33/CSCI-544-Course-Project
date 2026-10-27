"""Pilot analysis: per-verifier metrics, the PLAN.md §7 gates, and an exploratory pair table.

    python -m analysis.pilot --bank bank/pilot/pilot_bank.jsonl --labels bank/pilot/pilot_labels.jsonl \\
        --scores results/pilot/ar results/pilot/llada results/pilot/dream --out results/pilot/analysis

--scores takes score files (*.jsonl) or folders searched for them (AR scores_A/B.jsonl and
diffusion pilot_T*.jsonl). Each row names its model and template, so files can come in
any order. T3 asks whether the answer is *incorrect*: the AR harness already flips its
sign, the diffusion scorer does not, so diffusion T3 scores are flipped here.

What it writes (all numbers are pilot numbers: exploratory, PLAN.md §2):
- metrics.json and verifiers.csv: AUROC with a 95% problem-bootstrap interval, ECE,
  median Yes+No mass, warm seconds per candidate, within-pair ranking accuracy, per
  verifier, template and domain;
- gates.json and gates.md: usable scores, gate A (competence, with diffusion models within
  0.10 AUROC of Llama-3.2-3B), gate B (T1-T3 AUROC spread <= 0.05), dynamic range, no
  shortcuts, and the code rule;
- pairs.csv: every eligible pair at 90% pair TAR (AND rule), with FAR, its interval, kappa
  on false accepts and Spearman on wrong answers. Thresholds are set on the pilot itself,
  so these FARs are in-sample and are not the H1 test, which needs frozen validation
  thresholds (PLAN.md §5);
- figures: AUROC by verifier, template stability, pair FAR, overlap against competence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np

from analysis.metrics import (ProblemBootstrap, auroc, cohen_kappa, ece, pair_thresholds, spearman,
                              threshold_at_tar)

# model id or short key -> (display name, family, class, role)
VERIFIERS = {
    "Qwen/Qwen2.5-7B-Instruct": ("Qwen2.5-7B", "qwen", "AR", "primary"),
    "mistralai/Mistral-7B-Instruct-v0.3": ("Mistral-7B", "mistral", "AR", "primary"),
    "meta-llama/Llama-3.1-8B-Instruct": ("Llama-3.1-8B", "llama", "AR", "primary"),
    "google/gemma-2-9b-it": ("Gemma-2-9B", "gemma", "AR", "primary"),
    "meta-llama/Llama-3.2-3B-Instruct": ("Llama-3.2-3B", "llama", "AR", "small"),
    "Qwen/Qwen2.5-1.5B-Instruct": ("Qwen2.5-1.5B", "qwen", "AR", "small"),
    "llada": ("LLaDA-8B", "llada", "diffusion", "primary"),
    "GSAI-ML/LLaDA-8B-Instruct": ("LLaDA-8B", "llada", "diffusion", "primary"),
    "dream": ("Dream-7B", "dream", "diffusion", "primary"),
    "illada": ("iLLaDA-8B", "illada", "diffusion", "primary"),
}
SMALL_REFERENCE = "Llama-3.2-3B"
DOMAINS = ("math", "code", "pooled")
TEMPLATES = ("T1", "T2", "T3")
GATE_A_MIN, GATE_A_SMALL_GAP, GATE_B_SPREAD = 0.60, 0.10, 0.05
PAIR_TAR = 0.90
HEDGES = re.compile(r"\b(maybe|perhaps|might|assum\w*|approximately|i think|not sure|likely|seems)\b", re.I)

AR_COLOR, DIFF_COLOR = "#2a78d6", "#eb6834"   # dataviz reference palette, categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- loading

def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_files(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(p.rglob("scores_[AB].jsonl")) + sorted(p.rglob("pilot_T*.jsonl"))
        else:
            out.append(p)
    return out


def load_scores(files: list[Path], index: dict[str, int]) -> dict:
    """(verifier, template) -> {"score", "mass", "seconds"} arrays aligned to the bank (nan = missing)."""
    out = {}
    for path in files:
        for r in read_jsonl(path):
            key = r.get("model_id") or r.get("model")
            if key not in VERIFIERS:
                raise SystemExit(f"{path}: unknown model {key!r}")
            name, _, kind, _ = VERIFIERS[key]
            cid = r.get("candidate_id") or r.get("id")
            if cid not in index:
                continue
            slot = out.setdefault((name, r["template"]), {
                k: np.full(len(index), np.nan) for k in ("score", "mass", "seconds")})
            if r.get("status") not in ("ok", "completed") or r.get("score") is None:
                continue
            score = float(r["score"])
            if kind == "diffusion" and r["template"] == "T3":
                score = -score
            i = index[cid]
            slot["score"][i] = score
            slot["mass"][i] = r.get("yes_no_mass", np.nan)
            seconds = r.get("runtime_s") if "runtime_s" in r else (None if r.get("cold") else r.get("forward_seconds"))
            slot["seconds"][i] = np.nan if seconds is None else seconds
    return out


def meta(name: str) -> tuple[str, str, str]:
    for display, family, kind, role in VERIFIERS.values():
        if display == name:
            return family, kind, role
    raise KeyError(name)


# ---------------------------------------------------------------- per-verifier metrics

def domain_mask(bank: list[dict], domain: str) -> np.ndarray:
    return np.array([domain == "pooled" or r["domain"] == domain for r in bank])


def pair_ranking(bank, labels, scores, mask) -> float:
    by_pair = {}
    for i, r in enumerate(bank):
        if mask[i] and np.isfinite(scores[i]):
            by_pair.setdefault(r["pair_id"], {})[int(labels[i])] = scores[i]
    full = [d for d in by_pair.values() if len(d) == 2]
    if not full:
        return float("nan")
    return float(np.mean([(d[1] > d[0]) + 0.5 * (d[1] == d[0]) for d in full]))


def verifier_metrics(bank, labels, scores, n_boot) -> list[dict]:
    rows = []
    problems = [r["problem_id"] for r in bank]
    for (name, template), s in sorted(scores.items()):
        family, kind, role = meta(name)
        for domain in DOMAINS:
            m = domain_mask(bank, domain) & np.isfinite(s["score"])
            if m.sum() == 0:
                continue
            idx = np.flatnonzero(m)
            x, y = s["score"][idx], labels[idx]
            boot = ProblemBootstrap([problems[i] for i in idx], n=n_boot)
            lo, hi = boot.interval(lambda ix: auroc(x[ix], y[ix]))
            secs = s["seconds"][idx]
            rows.append({
                "verifier": name, "class": kind, "role": role, "template": template, "domain": domain,
                "n": int(m.sum()), "coverage": float(m.sum() / domain_mask(bank, domain).sum()),
                "auroc": auroc(x, y), "auroc_lo": lo, "auroc_hi": hi,
                "ece": ece(x, y), "median_yes_no_mass": float(np.nanmedian(s["mass"][idx])),
                "pair_ranking_accuracy": pair_ranking(bank, labels, s["score"], m),
                "seconds_per_candidate": float(np.nanmean(secs)) if np.isfinite(secs).any() else float("nan"),
            })
    return rows


def lookup(rows, verifier, template, domain):
    for r in rows:
        if (r["verifier"], r["template"], r["domain"]) == (verifier, template, domain):
            return r
    return None


# ---------------------------------------------------------------- shortcut scores

def surface_features(text: str) -> list[float]:
    return [math.log1p(len(text)), text.count("\n"), len(re.findall(r"\d+(?:\.\d+)?", text)),
            len(HEDGES.findall(text))]


def fit_logistic(X: np.ndarray, y: np.ndarray, steps: int = 3000, lr: float = 0.1) -> tuple:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + 1e-3 * np.r_[0, w[1:]])
    return w, mu, sd


def shortcut_check(bank, labels, dev_bank, dev_labels) -> dict:
    out = {}
    for domain in ("math", "code"):
        m = domain_mask(bank, domain)
        if not m.any():
            continue
        lengths = np.array([len(r["candidate"]) for r in bank])[m]
        a = auroc(lengths, labels[m])
        entry = {"length_auroc": max(a, 1 - a)}
        dev = [(r, dev_labels[r["candidate_id"]]) for r in dev_bank if r["domain"] == domain] if dev_bank else []
        if dev:
            w, mu, sd = fit_logistic(np.array([surface_features(r["candidate"]) for r, _ in dev]),
                                     np.array([l for _, l in dev], dtype=float))
            X = np.array([surface_features(r["candidate"]) for r in bank])[m]
            z = np.c_[np.ones(len(X)), (X - mu) / sd] @ w
            entry["surface_auroc"] = auroc(z, labels[m])
            entry["surface_fit_on"] = f"development bank ({len(dev)} {domain} answers)"
        else:
            entry["surface_auroc"] = None
            entry["surface_fit_on"] = f"no {domain} development answers yet"
        entry["passes"] = entry["length_auroc"] <= 0.65 and (entry["surface_auroc"] is None or entry["surface_auroc"] <= 0.65)
        out[domain] = entry
    return out


# ---------------------------------------------------------------- gates

def gates(rows, bank, labels, scores, shortcuts) -> dict:
    verifiers = sorted({r["verifier"] for r in rows})
    out = {"usable_scores": {}, "gate_a": {}, "gate_b": {}, "dynamic_range": {}, "no_shortcuts": shortcuts}
    for v in verifiers:
        t1 = lookup(rows, v, "T1", "pooled")
        if t1:
            out["usable_scores"][v] = {"median_yes_no_mass": t1["median_yes_no_mass"],
                                       "passes": t1["median_yes_no_mass"] >= 0.5}
    for domain in ("math", "code"):
        small = lookup(rows, SMALL_REFERENCE, "T1", domain)
        for v in verifiers:
            r = lookup(rows, v, "T1", domain)
            if r is None:
                continue
            _, kind, role = meta(v)
            ok = r["auroc"] >= GATE_A_MIN and r["auroc_lo"] > 0.5
            entry = {"auroc": r["auroc"], "ci": [r["auroc_lo"], r["auroc_hi"]], "class": kind, "role": role}
            if kind == "diffusion":
                if small is None:
                    entry["small_reference"] = None
                    entry["note"] = f"{SMALL_REFERENCE} not scored; the 0.10 rule is unchecked"
                else:
                    entry["small_reference"] = small["auroc"]
                    ok = ok and r["auroc"] >= small["auroc"] - GATE_A_SMALL_GAP
            entry["passes"] = bool(ok)
            out["gate_a"].setdefault(v, {})[domain] = entry
        for v in verifiers:
            vals = {t: lookup(rows, v, t, domain) for t in TEMPLATES}
            if all(vals.values()):
                aucs = {t: vals[t]["auroc"] for t in TEMPLATES}
                spread = max(aucs.values()) - min(aucs.values())
                out["gate_b"].setdefault(v, {})[domain] = {"aurocs": aucs, "spread": spread,
                                                           "passes": spread <= GATE_B_SPREAD}
        # dynamic range: the best primary AR verifier's FAR at its own 90% TAR threshold
        best = max((lookup(rows, v, "T1", domain) for v in verifiers
                    if meta(v)[1:] == ("AR", "primary") and lookup(rows, v, "T1", domain)),
                   key=lambda r: r["auroc"], default=None)
        if best:
            s = scores[(best["verifier"], "T1")]["score"]
            m = domain_mask(bank, domain) & np.isfinite(s)
            t = threshold_at_tar(s[m & (labels == 1)], PAIR_TAR)
            far = float(np.mean(s[m & (labels == 0)] >= t))
            out["dynamic_range"][domain] = {"verifier": best["verifier"], "far_at_90_tar": far,
                                            "passes": 0.10 <= far <= 0.70, "target_band": [0.15, 0.60]}
    for v in list(out["gate_b"]):
        pooled = lookup(rows, v, "T1", "pooled"), lookup(rows, v, "T2", "pooled"), lookup(rows, v, "T3", "pooled")
        if all(pooled):
            aucs = [p["auroc"] for p in pooled]
            out["gate_b"][v]["pooled"] = {"aurocs": dict(zip(TEMPLATES, aucs)), "spread": max(aucs) - min(aucs),
                                          "passes": max(aucs) - min(aucs) <= GATE_B_SPREAD}
    ar_code = [v for v, d in out["gate_a"].items() if meta(v)[1:] == ("AR", "primary") and d.get("code", {}).get("passes")]
    diff_code = [v for v, d in out["gate_a"].items() if meta(v)[1] == "diffusion" and d.get("code", {}).get("passes")]
    out["code_rule"] = {"ar_passing_code": ar_code, "diffusion_passing_code": diff_code,
                        "code_in_primary": len(ar_code) >= 2 and len(diff_code) >= 1}
    eligible = {}
    domains = ["math"] + (["code"] if out["code_rule"]["code_in_primary"] else [])
    for v, by_domain in out["gate_a"].items():
        if meta(v)[2] != "primary":
            continue
        a_ok = all(by_domain.get(d, {}).get("passes") for d in domains)
        b = out["gate_b"].get(v)
        b_ok = None if b is None else all(b.get(d, {}).get("passes", False) for d in domains)
        u_ok = out["usable_scores"].get(v, {}).get("passes", False)
        eligible[v] = {"usable": u_ok, "gate_a": a_ok, "gate_b": b_ok,
                       "eligible": bool(u_ok and a_ok and b_ok is not False)}
    out["eligibility"] = {"primary_domains": domains, "verifiers": eligible}
    diffusion_ok = [v for v, e in eligible.items() if meta(v)[1] == "diffusion" and e["eligible"]]
    out["decision"] = ("continue: a diffusion verifier passes the gates" if diffusion_ok
                       else "repair: no diffusion verifier passes; try the reconstruction score (PLAN.md §7)")
    return out


# ---------------------------------------------------------------- pairs

def pair_table(bank, labels, scores, names, n_boot) -> list[dict]:
    rows = []
    problems = [r["problem_id"] for r in bank]
    for a, b in combinations(sorted(names), 2):
        fa, ka, ra = meta(a)
        fb, kb, rb = meta(b)
        if fa == fb or (ka == kb == "diffusion"):
            continue
        if ra != "primary" or rb != "primary":
            continue
        kind = "AR+AR" if ka == kb == "AR" else "AR+diffusion"
        note = "E1 shared lineage (not in H1)" if {fa, fb} == {"qwen", "dream"} else ""
        for domain in ("math", "code", "pooled"):
            sa, sb = scores[(a, "T1")]["score"], scores[(b, "T1")]["score"]
            m = domain_mask(bank, domain) & np.isfinite(sa) & np.isfinite(sb)
            if domain == "pooled":
                # thresholds stay per domain (PLAN.md §5 step 2); pooling just counts every wrong answer
                acc = np.zeros(len(bank), dtype=bool)
                ok = True
                for d in ("math", "code"):
                    md = domain_mask(bank, d) & m
                    if not (md & (labels == 1)).any():
                        ok = False
                        break
                    ta, tb, _ = pair_thresholds(sa[md & (labels == 1)], sb[md & (labels == 1)], PAIR_TAR)
                    acc |= md & (sa >= ta) & (sb >= tb)
                if not ok:
                    continue
                r_ind = float("nan")
            else:
                if not (m & (labels == 1)).any() or not (m & (labels == 0)).any():
                    continue
                ta, tb, r_ind = pair_thresholds(sa[m & (labels == 1)], sb[m & (labels == 1)], PAIR_TAR)
                acc = m & (sa >= ta) & (sb >= tb)
            idx = np.flatnonzero(m)
            wrong = idx[labels[idx] == 0]
            boot = ProblemBootstrap([problems[i] for i in idx], n=n_boot)
            far_lo, far_hi = boot.interval(lambda ix: acc[idx[ix]][labels[idx[ix]] == 0].mean())
            if domain == "pooled":
                accept_a = accept_b = None
            else:
                accept_a, accept_b = sa[wrong] >= ta, sb[wrong] >= tb
            rows.append({
                "pair": f"{a} + {b}", "a": a, "b": b, "kind": kind, "note": note, "domain": domain,
                "n_correct": int((labels[idx] == 1).sum()), "n_wrong": int(len(wrong)),
                "individual_tar": r_ind, "pair_tar": float(acc[idx][labels[idx] == 1].mean()),
                "far": float(acc[wrong].mean()), "far_lo": far_lo, "far_hi": far_hi,
                "kappa_false_accepts": cohen_kappa(accept_a, accept_b) if accept_a is not None else float("nan"),
                "spearman_wrong": spearman(sa[wrong], sb[wrong]) if domain != "pooled" else float("nan"),
            })
    return rows


# ---------------------------------------------------------------- figures

def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figures(rows, pair_rows, gate, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    written = []
    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=AR_COLOR, markersize=8, label="AR"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor=DIFF_COLOR, markersize=8, label="Diffusion")]
    domains = [d for d in ("math", "code") if any(r["domain"] == d for r in rows)]

    # 1. AUROC by verifier (T1), per domain, with problem-bootstrap intervals
    fig, axes = plt.subplots(1, len(domains), figsize=(5.2 * len(domains), 3.6), sharey=True, squeeze=False)
    for ax, domain in zip(axes[0], domains):
        rs = sorted((r for r in rows if r["template"] == "T1" and r["domain"] == domain), key=lambda r: r["auroc"])
        y = np.arange(len(rs))
        for yi, r in zip(y, rs):
            c = DIFF_COLOR if r["class"] == "diffusion" else AR_COLOR
            ax.plot([r["auroc_lo"], r["auroc_hi"]], [yi, yi], color=c, linewidth=2, solid_capstyle="round")
            ax.plot(r["auroc"], yi, "o", color=c, markersize=8, markeredgecolor="white", markeredgewidth=2)
            ax.text(r["auroc_hi"] + 0.012, yi, f"{r['auroc']:.2f}", va="center", fontsize=8, color=MUTED)
        ax.axvline(GATE_A_MIN, color=MUTED, linestyle="--", linewidth=1)
        ax.text(GATE_A_MIN + 0.005, -0.6, "gate A 0.60", fontsize=8, color=MUTED, va="bottom")
        ax.set_yticks(y, [r["verifier"] for r in rs])
        ax.set_xlim(0.4, 1.05)
        ax.set_title(f"{domain.capitalize()}: AUROC (T1), 95% CI", fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0][-1].legend(handles=legend, frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "auroc_by_verifier.png", dpi=200)
    plt.close(fig)
    written.append("auroc_by_verifier.png")

    # 2. Template stability (gate B): T1-T3 range per verifier
    if gate["gate_b"]:
        fig, axes = plt.subplots(1, len(domains), figsize=(5.2 * len(domains), 3.6), sharey=True, squeeze=False)
        for ax, domain in zip(axes[0], domains):
            items = sorted(((v, d[domain]) for v, d in gate["gate_b"].items() if domain in d), key=lambda x: x[1]["spread"])
            for yi, (v, e) in enumerate(items):
                c = DIFF_COLOR if meta(v)[1] == "diffusion" else AR_COLOR
                vals = list(e["aurocs"].values())
                ax.plot([min(vals), max(vals)], [yi, yi], color=c, linewidth=2, solid_capstyle="round")
                for t, val in e["aurocs"].items():
                    ax.plot(val, yi, "o", color=c, markersize=6, markeredgecolor="white", markeredgewidth=1.5)
                ax.text(max(vals) + 0.012, yi, f"spread {e['spread']:.3f}" + ("" if e["passes"] else "  ✗"),
                        va="center", fontsize=8, color=MUTED)
            ax.set_yticks(range(len(items)), [v for v, _ in items])
            ax.set_xlim(0.4, 1.1)
            ax.set_title(f"{domain.capitalize()}: AUROC across T1–T3 (gate B ≤ 0.05)", fontsize=10, color=INK, loc="left")
            _style(ax)
        fig.tight_layout()
        fig.savefig(out / "template_stability.png", dpi=200)
        plt.close(fig)
        written.append("template_stability.png")

    # 3. Pair FAR at 90% pair TAR
    for domain in domains + ["pooled"]:
        prs = sorted((p for p in pair_rows if p["domain"] == domain), key=lambda p: -p["far"])
        if not prs:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(prs) + 1.2))
        for yi, p in enumerate(prs):
            c = DIFF_COLOR if p["kind"] == "AR+diffusion" else AR_COLOR
            ax.plot([p["far_lo"], p["far_hi"]], [yi, yi], color=c, linewidth=2, solid_capstyle="round", alpha=0.6)
            ax.plot(p["far"], yi, "o", color=c, markersize=7, markeredgecolor="white", markeredgewidth=2)
        ax.set_yticks(range(len(prs)), [p["pair"] + (" (E1)" if p["note"] else "") for p in prs], fontsize=8)
        ax.set_xlabel("False-accept rate at 90% pair TAR (lower is better)", fontsize=9, color=INK)
        ax.set_title(f"{domain.capitalize()}: pair FAR, AND rule, in-sample pilot thresholds", fontsize=10, color=INK, loc="left")
        ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=AR_COLOR, markersize=8, label="AR+AR"),
                           Line2D([0], [0], marker="o", color="w", markerfacecolor=DIFF_COLOR, markersize=8, label="AR+diffusion")],
                  frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        _style(ax)
        fig.tight_layout()
        fig.savefig(out / f"pair_far_{domain}.png", dpi=200)
        plt.close(fig)
        written.append(f"pair_far_{domain}.png")

    # 4. H2 preview: kappa on false accepts against the pair's mean AUROC
    for domain in domains:
        prs = [p for p in pair_rows if p["domain"] == domain and np.isfinite(p["kappa_false_accepts"])]
        if not prs:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for p in prs:
            mean_auc = np.mean([lookup(rows, p["a"], "T1", domain)["auroc"], lookup(rows, p["b"], "T1", domain)["auroc"]])
            c = DIFF_COLOR if p["kind"] == "AR+diffusion" else AR_COLOR
            ax.plot(mean_auc, p["kappa_false_accepts"], "o", color=c, markersize=8, markeredgecolor="white", markeredgewidth=2)
        ax.set_xlabel("Mean AUROC of the pair (T1)", fontsize=9, color=INK)
        ax.set_ylabel("κ on false accepts", fontsize=9, color=INK)
        ax.set_title(f"{domain.capitalize()}: error overlap vs competence (H2 preview)", fontsize=10, color=INK, loc="left")
        ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=AR_COLOR, markersize=8, label="AR+AR"),
                           Line2D([0], [0], marker="o", color="w", markerfacecolor=DIFF_COLOR, markersize=8, label="AR+diffusion")],
                  frameon=False, fontsize=8)
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        fig.tight_layout()
        fig.savefig(out / f"overlap_vs_competence_{domain}.png", dpi=200)
        plt.close(fig)
        written.append(f"overlap_vs_competence_{domain}.png")
    return written


# ---------------------------------------------------------------- report

def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def markdown(rows, pair_rows, gate, bank_info) -> str:
    L = ["# Pilot results", "", f"Bank: {bank_info}", "",
         "All numbers are pilot (exploratory) numbers. Pair thresholds are set on the pilot itself, "
         "so pair FARs are in-sample and are not the H1 test.", "",
         "## Verifiers (T1)", "", "| Verifier | Class | Domain | AUROC [95% CI] | ECE | Yes+No mass | Pair rank acc. | s / candidate |",
         "|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in rows if r["template"] == "T1"), key=lambda r: (r["domain"], -r["auroc"])):
        L.append(f"| {r['verifier']} | {r['class']} | {r['domain']} | {fmt(r['auroc'])} [{fmt(r['auroc_lo'])}, {fmt(r['auroc_hi'])}] "
                 f"| {fmt(r['ece'])} | {fmt(r['median_yes_no_mass'])} | {fmt(r['pair_ranking_accuracy'])} | {fmt(r['seconds_per_candidate'])} |")
    L += ["", "## Gates (PLAN.md §7)", ""]
    L.append("| Verifier | Usable scores | Gate A math | Gate A code | Gate B math | Gate B code | Eligible |")
    L.append("|---|---|---|---|---|---|---|")
    yes = lambda b: "–" if b is None else ("pass" if b else "**fail**")
    star = lambda e: "*" if e and e.get("class") == "diffusion" and e.get("small_reference") is None else ""
    for v in sorted(gate["gate_a"]):
        ga, gb = gate["gate_a"][v], gate["gate_b"].get(v, {})
        e = gate["eligibility"]["verifiers"].get(v, {})
        L.append(f"| {v} | {yes(gate['usable_scores'].get(v, {}).get('passes'))} "
                 f"| {yes(ga.get('math', {}).get('passes'))}{star(ga.get('math'))} | {yes(ga.get('code', {}).get('passes'))}{star(ga.get('code'))} "
                 f"| {yes(gb.get('math', {}).get('passes'))} | {yes(gb.get('code', {}).get('passes'))} "
                 f"| {'reference only' if meta(v)[2] != 'primary' else yes(e.get('eligible'))} |")
    L += ["", f"\* {SMALL_REFERENCE} not scored on this bank, so the diffusion 'within 0.10 of the small AR verifier' part of gate A is unchecked.",
          f"- **Code rule:** {gate['code_rule']}",
          f"- **Dynamic range:** " + "; ".join(f"{d}: {e['verifier']} FAR {fmt(e['far_at_90_tar'])} at 90% TAR "
                                              f"({'pass' if e['passes'] else 'fail'})" for d, e in gate['dynamic_range'].items()),
          f"- **No shortcuts:** " + "; ".join(f"{d}: length AUROC {fmt(e['length_auroc'])}, surface AUROC {fmt(e['surface_auroc'])} "
                                             f"({'pass' if e['passes'] else 'fail'})" for d, e in gate['no_shortcuts'].items()),
          f"- **Decision:** {gate['decision']}", "",
          "## Pairs at 90% pair TAR (AND rule, T1)", "",
          "| Pair | Kind | Domain | FAR [95% CI] | Pair TAR | κ false accepts | Spearman (wrong) |", "|---|---|---|---|---|---|---|"]
    for p in sorted(pair_rows, key=lambda p: (p["domain"], p["far"])):
        L.append(f"| {p['pair']}{' (E1)' if p['note'] else ''} | {p['kind']} | {p['domain']} | {fmt(p['far'])} "
                 f"[{fmt(p['far_lo'])}, {fmt(p['far_hi'])}] | {fmt(p['pair_tar'])} | {fmt(p['kappa_false_accepts'])} | {fmt(p['spearman_wrong'])} |")
    for domain in ("math", "code", "pooled"):
        h1 = [p for p in pair_rows if p["domain"] == domain and not p["note"]]
        best_ar = min((p for p in h1 if p["kind"] == "AR+AR"), key=lambda p: p["far"], default=None)
        best_mix = min((p for p in h1 if p["kind"] == "AR+diffusion"), key=lambda p: p["far"], default=None)
        if best_ar and best_mix:
            L.append(f"\n**{domain}:** best AR+AR {best_ar['pair']} FAR {fmt(best_ar['far'])}; best AR+diffusion "
                     f"{best_mix['pair']} FAR {fmt(best_mix['far'])}; ΔFAR {best_mix['far'] - best_ar['far']:+.3f} (in-sample, exploratory).")
    return "\n".join(L) + "\n"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-bank", default="bank/dev/dev_bank.jsonl", help="for fitting the surface-feature shortcut score")
    ap.add_argument("--dev-labels", default="bank/dev/dev_labels.jsonl")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bank = read_jsonl(args.bank)
    label_of = {r["candidate_id"]: r["label"] for r in read_jsonl(args.labels)}
    labels = np.array([label_of[r["candidate_id"]] for r in bank], dtype=float)
    index = {r["candidate_id"]: i for i, r in enumerate(bank)}
    scores = load_scores(score_files(args.scores), index)
    names = sorted({v for v, t in scores if t == "T1"})

    rows = verifier_metrics(bank, labels, scores, args.bootstrap)
    dev_bank = read_jsonl(args.dev_bank) if Path(args.dev_bank).is_file() else []
    dev_labels = {r["candidate_id"]: r["label"] for r in read_jsonl(args.dev_labels)} if dev_bank else {}
    gate = gates(rows, bank, labels, scores, shortcut_check(bank, labels, dev_bank, dev_labels))
    pair_rows = pair_table(bank, labels, scores, names, args.bootstrap)

    import hashlib
    bank_hash = hashlib.sha256(Path(args.bank).read_bytes()).hexdigest()
    counts = {d: {"correct": int(((labels == 1) & domain_mask(bank, d)).sum()),
                  "wrong": int(((labels == 0) & domain_mask(bank, d)).sum())} for d in ("math", "code")}
    bank_info = f"`{args.bank}` (sha256 {bank_hash[:12]}…), " + ", ".join(
        f"{d}: {c['correct']} correct / {c['wrong']} wrong" for d, c in counts.items() if c["correct"] + c["wrong"])
    (out / "metrics.json").write_text(json.dumps({"bank_sha256": bank_hash, "counts": counts, "verifiers": rows,
                                                  "pairs": pair_rows}, indent=1, default=float) + "\n",
                                      encoding="utf-8", newline="\n")
    (out / "gates.json").write_text(json.dumps(gate, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    write_csv(out / "verifiers.csv", rows)
    write_csv(out / "pairs.csv", pair_rows)
    (out / "gates.md").write_text(markdown(rows, pair_rows, gate, bank_info), encoding="utf-8", newline="\n")
    written = figures(rows, pair_rows, gate, out)
    print(f"{len(rows)} verifier rows, {len(pair_rows)} pair rows; figures: {', '.join(written)}")
    print(gate["decision"])


if __name__ == "__main__":
    main()
