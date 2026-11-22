"""Tables and figures from a results.json written by analysis.final_evaluate (or analysis.final_pilot).

    python -m analysis.final_report --results results/final/test/results.json --out results/final/test

Computes nothing: every number is read from results.json, so the report can be rebuilt without touching labels.
Writes summary.md (the tables), CSV files and PNG figures. Results from the pilot carry their status into every
title, because they are in-sample and are never H1 results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.final_stats import VERDICTS

AR_COLOR, DIFF_COLOR = "#2a78d6", "#eb6834"     # same two hues as the pilot figures
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GROUP_NAME = {"ar_ar": "AR+AR", "ar_diffusion": "AR+diffusion"}


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def ci(v, nd=3):
    return "–" if v is None else f"[{fmt(float(v[0]), nd)}, {fmt(float(v[1]), nd)}]"


def points(d: dict) -> str:
    return f"{d['points']:+.1f} points, 95% {ci(d['ci95'], 1)}, 90% {ci(d['ci90'], 1)}"


def color(kind: str) -> str:
    return DIFF_COLOR if kind == "AR+diffusion" or kind == "diffusion" else AR_COLOR if kind in ("AR+AR", "AR") else MUTED


def short(r: dict) -> str:
    """Group label for a figure: generators without their organization, sources without their split."""
    parts = str(r["group"]).split("/")
    return parts[-1] if r["grouping"] == "generator" else parts[0] if parts[-1] == "test" else parts[-1]


def is_pilot(results: dict) -> bool:
    return "pilot" in results["status"].lower()


# ---------------------------------------------------------------- tables

def counts(r: dict) -> str:
    return (f"{fmt(r['tar'])} ({r['accepted_correct']}/{r['n_correct']}) {ci(r['tar_ci'])} | "
            f"{fmt(r['far'])} ({r['accepted_wrong']}/{r['n_wrong']}) {ci(r['far_ci'])}")


def h1_tables(results: dict) -> list[str]:
    h1 = results["h1"]
    main = h1["by_tar"]["0.90"]
    name = "Decision-table row the code returns (smoke test of the code, not an H1 result)" if is_pilot(results) else "Verdict"
    L = [f"## Final test: {h1['test_name']} ({h1['domain']})", "",
         f"**{name}:** {h1['verdict_text']}", "",
         "| Row | Pair or verifier | TAR (counts) [95% CI] | FAR (counts) [95% CI] |", "|---|---|---|---|"]
    for g, r in main["rows"].items():
        L.append(f"| selected {GROUP_NAME[g]} | {h1['pairs'][g]} | {counts(r)} |")
    for r in results["reference_rows"]:
        mass = f" (T2 median Yes+No mass {r['median_mass_t2']:.2f})" if "median_mass_t2" in r else ""
        L.append(f"| {r['row']} | {r['name']}{mass} | {counts(r)} |")
    ideal = h1["ideal_thresholds_test_tuned"]
    for g, r in ideal["rows"].items():
        L.append(f"| ideal thresholds (test-tuned), {GROUP_NAME[g]} | {h1['pairs'][g]} | {counts(r)} |")
    m, old, b = h1["matching"], h1["matching_old_rule"], h1["selection_bootstrap"]
    L += ["", f"- **ΔFAR = FAR(AR+diffusion) − FAR(AR+AR):** {points(main['delta_far'])}",
          f"- **Smallest effect these data could have detected** (80% power, 5% two-sided): {h1['detectable_points']:.1f} points",
          f"- **Matching check:** off 90% by {', '.join(f'{v:.1f}' for v in m['off_target_points'])} points "
          f"(limit {m['limits_points']['off_target']:.0f}), gap {m['gap_points']:.1f} points (limit {m['limits_points']['gap']:.0f}, "
          f"95% {ci(main['tar_gap_ci'], 1)}): " + ("passes" if m["passes"] else "**fails**"),
          f"- **Matching check under the rule before 2026-11-21** (3 points and 3 points; reported beside it, changes nothing): "
          + ("passes" if old["passes"] else "fails") + f", which would give: {VERDICTS[old['row']]}",
          f"- **Ideal thresholds (test-tuned; no verdict uses it):** ΔFAR {points(ideal['delta_far'])}",
          *(f"- **{GROUP_NAME[g]} pair against the best single verifier:** FAR difference {points(d)}"
            for g, d in results["reference_rows"][0]["pair_minus_single"].items()),
          f"- **Selection repeated in the bootstrap (secondary):** choice only, 95% {ci(b['choice_only']['ci95'], 1)}; choice and "
          f"thresholds, 95% {ci(b['choice_and_thresholds']['ci95'], 1)} points",
          f"- **LLaDA below gate A's bar on validation:** {h1['llada_below_bar_on_validation']}; "
          f"**comparable-cost claim:** {h1['comparable_cost_claim']}"]
    for g, p in h1["pseudo_check"].items():
        if p["available"]:
            L.append(f"- **{GROUP_NAME[g]} pair against the pseudo-pair of its stronger member ({p['stronger']}):** "
                     f"FAR difference {points(p['delta_far'])}; TAR {fmt(p['pair_tar'])} and {fmt(p['pseudo_tar'])} "
                     f"({'matched' if p['matching']['passes'] else 'not matched'}); beats it: {p['beats']}")
        else:
            L.append(f"- **{GROUP_NAME[g]} pair:** its stronger member ({p['stronger']}) has no second wording, so no pseudo-pair")
    if h1["budget_matched"]:
        L.append(f"- **Budget-matched (secondary):** the cheaper {GROUP_NAME[h1['budget_matched']['cheaper']]} pair averages T1 and T2: "
                 f"ΔFAR {points(h1['budget_matched']['delta_far'])}")
    r = h1["reading"]
    L += ["", f"**Reading (PLAN.md §8):** H2 for the selected pair is *{r['h2']}*. {r['reading']}. {r['claim']}.", ""]
    L += [f"- Caveat: {c}" for c in r["caveats"]]
    L += ["", "| Pair TAR | ΔFAR (AR+diffusion − AR+AR) | TAR AR+diffusion / AR+AR |", "|---|---|---|"]
    for tar, e in h1["by_tar"].items():
        L.append(f"| {float(tar):.0%} | {points(e['delta_far'])} | {fmt(e['rows']['ar_diffusion']['tar'])} / {fmt(e['rows']['ar_ar']['tar'])} |")
    return L


def main_table(results: dict) -> list[dict]:
    """The rows of the main table for a CSV: the selected pairs, the reference rows and the test-tuned rows."""
    if not results["h1"]:
        return []
    h1 = results["h1"]
    rows = [{"row": f"selected {GROUP_NAME[g]}", "name": h1["pairs"][g], **r} for g, r in h1["by_tar"]["0.90"]["rows"].items()]
    rows += [{k: v for k, v in r.items() if k != "pair_minus_single"} for r in results["reference_rows"]]
    return rows + [{"row": f"ideal thresholds (test-tuned), {GROUP_NAME[g]}", "name": h1["pairs"][g], **r}
                   for g, r in h1["ideal_thresholds_test_tuned"]["rows"].items()]


def family_tables(results: dict) -> list[str]:
    L = ["## AR-family study (H1 is not run)", "", f"Reason: {results['settings']['reason']}", ""]
    for d, f in results["ar_family"].items():
        L += [f"**{d}:** best single verifier {f['best_single']}, FAR {fmt(f['best_single_far'])}; a perfect verifier accepts no wrong answer.", "",
              "| Pair | Selected on validation | TAR | FAR | κ false accepts | Share of the best single's false accepts removed |",
              "|---|---|---|---|---|---|"]
        for r in f["pairs"]:
            L.append(f"| {r['pair']} | {'yes' if r['selected'] else ''} | {fmt(r['tar'])} | {fmt(r['far'])} {ci(r['far_ci'])} | "
                     f"{fmt(r['kappa'])} {ci(r['kappa_ci'])} | {fmt(r['share_of_single_false_accepts_removed'], 2)} |")
        L.append("")
    return L


def secondary_tables(results: dict) -> list[str]:
    L = ["", "## Pairs at 90% pair TAR", "", "| Domain | Pair | Kind | Roles | TAR | FAR [95% CI] | s / candidate | GPU |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in results["pairs"] if r["target_tar"] == 0.9 and r["kind"] != "pseudo"), key=lambda r: (r["domain"], r["far"])):
        L.append(f"| {r['domain']} | {r['pair']} | {r['kind']} | {', '.join(r['roles'])} | {fmt(r['tar'])} | {fmt(r['far'])} {ci(r['far_ci'])} | {fmt(r['seconds'])} | {' + '.join(r['gpu_types'])} |")
    for r in results["pooled"]:
        if r["kind"] == "difference":
            L.append(f"\nPooled (weights {r['weights']}): {r['pair']}: {r['delta_far_points']:+.1f} points, 95% {ci(r['ci95'], 1)}.")
    if results.get("pooled_note"):
        L.append(f"\nPooled: {results['pooled_note']}.")
    L += ["", "## Single verifiers", "", "| Domain | Score | AUROC [95% CI] | ECE | Median Yes+No mass | FAR at 90% TAR | s / candidate | GPU |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in results["singles"] if r["column"] in ("T1", "T2", "recon")), key=lambda r: (r["domain"], r["column"], -r["auroc"])):
        L.append(f"| {r['domain']} | {r['label']} | {fmt(r['auroc'])} {ci(r['auroc_ci'])} | {fmt(r['ece'])} | {fmt(r['median_mass'])} | "
                 f"{fmt(r['at_90_tar']['far'] if r['at_90_tar'] else None)} | {fmt(r['seconds'])} | {r['gpu'] or '–'} |")
    for d, h in results["h2"].items():
        t = h["trend"]
        L += ["", f"## H2 ({d}): overlap on wrong answers", "",
              f"Trend over {t['n_ar_pairs']} cross-family AR+AR pairs: κ = {fmt(t['intercept'])} + {fmt(t['slope'])} × mean AUROC "
              f"(slope 95% {ci(t['slope_ci'])}). \"On the trend\" needs a 90% interval inside ±{t['on_trend_margin']:g}.", "",
              "| Pair | Kind | κ [95% CI] | Spearman | CAPA (wrong) | CAPA (all) | κ minus trend [95% CI] | Status |", "|---|---|---|---|---|---|---|---|"]
        trend = {r["pair"]: r for r in t["pairs"]}
        for o in sorted(h["overlap"], key=lambda o: (o["kind"], o["kappa"] if np.isfinite(o["kappa"]) else 9)):
            r = trend.get(o["pair"])
            L.append(f"| {o['pair']} | {o['kind']} | {fmt(o['kappa'])} {ci(o['kappa_ci'])} | {fmt(o['spearman'])} | {fmt(o['capa_wrong'])} | "
                     f"{fmt(o['capa_all'])} | " + (f"{fmt(r['residual'])} {ci(r['residual_ci'])} | {r['status'] or 'in the fit'} |" if r else "– | – |"))
        for value, e in h["by_error_type"].items():
            if e["trend"] is None:
                L.append(f"\nWithin error type *{value}*: {e['n_wrong']} wrong answers, too few to report a trend.")
                continue
            L += ["", f"Within error type *{value}* ({e['n_wrong']} wrong answers, descriptive):", "",
                  "| AR+diffusion pair | κ | κ minus trend [95% CI] | Status |", "|---|---|---|---|"]
            L += [f"| {r['pair']} | {fmt(r['kappa'])} | {fmt(r['residual'])} {ci(r['residual_ci'])} | {r['status']} |"
                  for r in e["trend"]["pairs"] if not r["in_fit"]]
    side = results.get("reconstruction_side")
    if side:
        L += ["", "## Reconstruction score, side experiment (drawn test answers, frozen validation thresholds)", ""]
        if "error" in side:
            L.append(f"Failed: {side['error']}. The main result above does not depend on it.")
        else:
            L += ["| Score | AUROC [95% CI] | TAR at 90% | FAR at 90% |", "|---|---|---|---|"]
            L += [f"| {name} | {fmt(r['auroc'])} [{fmt(r['auroc_lo'])}, {fmt(r['auroc_hi'])}] | {fmt(r['points']['0.90']['tar'])} | "
                  f"{fmt(r['points']['0.90']['far'])} |" for name, r in side["runs"].items()]
    for d, rows in results["e1"].items():
        L += ["", f"## E1 ({d}, descriptive): Qwen2.5-7B with each partner", "",
              "| Pair | FAR | FAR minus Qwen+Dream | κ | κ minus Qwen+Dream |", "|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['pair']} | {fmt(r['far'])} {ci(r['far_ci'])} | {fmt(r['far_minus_dream'])} {ci(r['far_minus_dream_ci'])} | "
                     f"{fmt(r['kappa'])} {ci(r['kappa_ci'])} | {fmt(r['kappa_minus_dream'])} {ci(r['kappa_minus_dream_ci'])} |")
    if results["breakdowns"]:
        L += ["", "## Breakdowns (descriptive; small groups marked *)", "", "| Domain | Group | Wrong | FAR AR+diffusion | FAR AR+AR | ΔFAR points [95% CI] | κ AR+diffusion / AR+AR |",
              "|---|---|---|---|---|---|---|"]
        for r in results["breakdowns"]:
            delta = f"{r['delta_far_points']:+.1f} {ci(r['ci95'], 1)}" if "delta_far_points" in r else "–"
            L.append(f"| {r['domain']} | {r['grouping']}: {r['group']}{'*' if r['small'] else ''} | {r['n_wrong']} | {fmt(r.get('far_ar_diffusion'))} | "
                     f"{fmt(r.get('far_ar_ar'))} | {delta} | {fmt(r.get('kappa_ar_diffusion'))} / {fmt(r.get('kappa_ar_ar'))} |")
    return L


def markdown(results: dict) -> str:
    L = ["# Final analysis", "", f"**Status: {results['status']}.** Bootstrap over whole problems, {results['rules']['bootstrap']:,} "
         f"replicates, seed {results['rules']['seed']}, percentile intervals, thresholds fixed.", ""]
    if is_pilot(results):
        L += ["> Pilot numbers: exploratory and in-sample (thresholds and pairs were chosen on the same answers). "
              "They only show that the code runs. They are not H1 results.", ""]
    L += h1_tables(results) if results["h1"] else family_tables(results)
    return "\n".join(L + secondary_tables(results)) + "\n"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: "; ".join(map(str, v)) if isinstance(v, (list, tuple)) else json.dumps(v) if isinstance(v, dict) else v
                        for k, v in r.items()})


# ---------------------------------------------------------------- figures

def style(ax, grid="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def dot(ax, x, y, c, marker="o", size=7):
    ax.plot(x, y, marker, color=c, markersize=size, markeredgecolor="white", markeredgewidth=1.5)


def figures(results: dict, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    written, prefix = [], "Pilot, in-sample: " if is_pilot(results) else ""
    handle = lambda label, c, marker="o": Line2D([0], [0], marker=marker, color="w", markerfacecolor=c, markersize=8, label=label)
    two = [handle("AR+AR", AR_COLOR), handle("AR+diffusion", DIFF_COLOR)]
    three = two + [handle("single verifier", MUTED, "s")]
    domains = list(results["h2"])
    pairs90 = [r for r in results["pairs"] if r["target_tar"] == 0.9]

    def save(fig, name):
        fig.savefig(out / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(name)

    def title(ax, text):
        ax.set_title(prefix + text, fontsize=10, color=INK, loc="left")

    # 1. The final test: the effect in points against the margin, at each operating point
    if results["h1"]:
        h1 = results["h1"]
        fig, ax = plt.subplots(figsize=(6.2, 2.4))
        rows = [(f"{float(t):.0%} pair TAR", e["delta_far"]) for t, e in h1["by_tar"].items()]
        rows.append(("90%, test-tuned thresholds", h1["ideal_thresholds_test_tuned"]["delta_far"]))
        for y, (label, d) in enumerate(reversed(rows)):
            c = INK if label.startswith("90% pair") else MUTED
            ax.plot(d["ci95"], [y, y], color=c, linewidth=1.5, solid_capstyle="round")
            ax.plot(d["ci90"], [y, y], color=c, linewidth=4, solid_capstyle="round", alpha=0.45)
            dot(ax, d["points"], y, c)
        ax.axvspan(-3, 3, color=GRID, alpha=0.6, zorder=0)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel("ΔFAR in points: AR+diffusion − AR+AR (negative favors AR+diffusion)\nthin line 95%, thick line 90% interval; band: ±3 points",
                      fontsize=8, color=INK)
        title(ax, f"Selected pairs, {h1['domain']}")
        style(ax)
        save(fig, "h1_delta_far.png")

        fig, ax = plt.subplots(figsize=(5.4, 4))
        for name, pts in h1["tradeoff"].items():
            c = DIFF_COLOR if "LLaDA" in name else AR_COLOR
            chosen = name in h1["pairs"].values()
            ax.plot([p["tar"] for p in pts], [p["far"] for p in pts], color=c, linewidth=2.2 if chosen else 0.9, alpha=1 if chosen else 0.45)
        ax.axvline(0.9, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xlabel("Pair TAR", fontsize=9, color=INK)
        ax.set_ylabel("Pair FAR", fontsize=9, color=INK)
        ax.legend(handles=[Line2D([0], [0], color=AR_COLOR, linewidth=2, label="AR+AR"),
                           Line2D([0], [0], color=DIFF_COLOR, linewidth=2, label="AR+diffusion")], frameon=False, fontsize=8)
        title(ax, "Trade-off at validation thresholds (bold: selected pairs)")
        style(ax, "both")
        save(fig, "tradeoff.png")

    for d in domains:
        # 2. FAR of every cross-model pair at 90% pair TAR
        rs = sorted((r for r in pairs90 if r["domain"] == d and r["kind"] != "pseudo" and "template_avg" not in r["roles"]),
                    key=lambda r: -r["far"])
        fig, ax = plt.subplots(figsize=(6.4, 0.26 * len(rs) + 1.1))
        for y, r in enumerate(rs):
            final = {"h1_ar", "h1_mixed", "ar_family"} & set(r["roles"])
            ax.plot(r["far_ci"], [y, y], color=color(r["kind"]), linewidth=2, solid_capstyle="round", alpha=0.8 if final else 0.35)
            dot(ax, r["far"], y, color(r["kind"]), size=7 if final else 5)
        ax.set_yticks(range(len(rs)), [r["pair"] + (" ●" if {"h1_ar", "h1_mixed"} & set(r["roles"]) else "") for r in rs], fontsize=7)
        ax.set_xlabel("False-accept rate at 90% pair TAR, 95% CI (● pair of the final test)", fontsize=8, color=INK)
        ax.legend(handles=two, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        title(ax, f"{d.capitalize()}: pairs")
        style(ax)
        save(fig, f"pairs_far_{d}.png")

        # 3. FAR against measured cost
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        one_gpu = [r for r in results["singles"] if r["domain"] == d and r["column"] in ("T1", "recon") and len(r["gpu_types"]) == 1]
        types = [r["gpu_types"][0] for r in one_gpu]
        gpu = max(sorted(set(types)), key=types.count) if types else "unknown"     # seconds on other GPU types are left out
        shown = [r for r in one_gpu if r["gpu_types"] == [gpu] and np.isfinite(r["seconds"])]
        for r in pairs90:
            if (r["domain"] == d and {"h1_ar", "h1_mixed", "ar_family", "e1"} & set(r["roles"]) and r["gpu_types"] == [gpu]
                    and np.isfinite(r["seconds"])):
                dot(ax, r["seconds"], r["far"], color(r["kind"]))
        for k, r in enumerate(sorted(shown, key=lambda r: (r["at_90_tar"]["far"], r["seconds"]))):
            dot(ax, r["seconds"], r["at_90_tar"]["far"], MUTED, "s", 6)
            ax.annotate(r["label"], (r["seconds"], r["at_90_tar"]["far"]), xytext=(5, 4 if k % 2 else -9),
                        textcoords="offset points", fontsize=6.5, color=MUTED)
        ax.margins(0.1)
        ax.set_xlabel(f"Measured GPU-seconds per candidate ({gpu} runs only)", fontsize=8, color=INK)
        ax.set_ylabel("FAR at 90% TAR", fontsize=9, color=INK)
        ax.legend(handles=three, frameon=False, fontsize=8)
        title(ax, f"{d.capitalize()}: accuracy against cost")
        style(ax, "both")
        save(fig, f"cost_{d}.png")

        # 4. Overlap against competence
        t = results["h2"][d]["trend"]
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        xs = np.array([r["mean_auroc"] for r in t["pairs"]])
        span = np.linspace(xs.min() - 0.01, xs.max() + 0.01, 2)
        if np.isfinite(t["slope"]):
            ax.plot(span, t["intercept"] + t["slope"] * span, color=MUTED, linewidth=1.2, linestyle="--")
        for r in t["pairs"]:
            final = {"h1_ar", "h1_mixed"} & set(r["roles"])
            dot(ax, r["mean_auroc"], r["kappa"], color(r["kind"]), size=8 if final else 5.5)
            if r["kind"] == "AR+diffusion" and (final or "e1" in r["roles"]):
                ax.annotate(r["pair"], (r["mean_auroc"], r["kappa"]), xytext=(5, 4), textcoords="offset points", fontsize=6.5, color=MUTED)
        ax.set_xlabel("Mean AUROC of the pair", fontsize=9, color=INK)
        ax.set_ylabel("κ on false accepts", fontsize=9, color=INK)
        ax.legend(handles=two + [Line2D([0], [0], color=MUTED, linestyle="--", label="trend over AR+AR pairs")], frameon=False, fontsize=8)
        title(ax, f"{d.capitalize()}: error overlap against competence")
        style(ax, "both")
        save(fig, f"overlap_vs_competence_{d}.png")

        # 5. ROC and reliability, one small panel per verifier
        rs = [r for r in results["singles"] if r["domain"] == d and r["roc"]]
        cols = min(4, len(rs))
        rows = int(np.ceil(len(rs) / cols))
        for what, name in (("roc", f"roc_{d}.png"), ("reliability", f"reliability_{d}.png")):
            fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.2 * rows), squeeze=False, sharex=True, sharey=True)
            for ax in axes.flat[len(rs):]:
                ax.axis("off")
            for ax, r in zip(axes.flat, rs):
                ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
                if what == "roc":
                    ax.plot(*np.array(r["roc"]).T, color=color(r["kind"]), linewidth=1.8)
                    ax.set_title(f"{r['label']}\nAUROC {r['auroc']:.2f}", fontsize=7.5, color=INK, loc="left")
                elif r["reliability"]:
                    ax.plot([b["confidence"] for b in r["reliability"]], [b["accuracy"] for b in r["reliability"]], "-o",
                            color=color(r["kind"]), linewidth=1.5, markersize=3.5)
                    ax.set_title(f"{r['label']}\nECE {r['ece']:.2f}", fontsize=7.5, color=INK, loc="left")
                style(ax, "both")
            fig.supxlabel("False-accept rate" if what == "roc" else "P(Yes) / (P(Yes) + P(No))", fontsize=8, color=INK)
            fig.supylabel("True-accept rate" if what == "roc" else "Share correct", fontsize=8, color=INK)
            fig.suptitle(prefix + f"{d.capitalize()}: {'ROC' if what == 'roc' else 'reliability'} by verifier", fontsize=10, color=INK, x=0.02, ha="left")
            fig.tight_layout()
            save(fig, name)

        # 6. ΔFAR by source, generator and error type
        rs = [r for r in results["breakdowns"] if r["domain"] == d and "delta_far_points" in r]
        if rs:
            fig, ax = plt.subplots(figsize=(6.2, 0.3 * len(rs) + 1.1))
            for y, r in enumerate(reversed(rs)):
                c = MUTED if r["small"] else INK
                ax.plot(r["ci95"], [y, y], color=c, linewidth=2, solid_capstyle="round", alpha=0.6)
                dot(ax, r["delta_far_points"], y, c, size=6)
            ax.axvspan(-3, 3, color=GRID, alpha=0.6, zorder=0)
            ax.axvline(0, color=MUTED, linewidth=1)
            ax.set_yticks(range(len(rs)), [f"{r['grouping']}: {short(r)} (n={r['n_wrong']})" for r in reversed(rs)], fontsize=7)
            ax.set_xlabel("ΔFAR in points (AR+diffusion − AR+AR), 95% CI; gray: small group", fontsize=8, color=INK)
            title(ax, f"{d.capitalize()}: breakdowns (descriptive)")
            style(ax)
            save(fig, f"breakdown_{d}.png")
    return written


def write_report(results: dict, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(markdown(results), encoding="utf-8", newline="\n")
    singles = [{k: v for k, v in r.items() if k not in ("roc", "reliability")} for r in results["singles"]]
    tables = {"main_table.csv": main_table(results), "pairs.csv": results["pairs"], "pooled.csv": results["pooled"], "singles.csv": singles, "breakdowns.csv": results["breakdowns"],
              "overlap.csv": [{"domain": d, **o} for d, h in results["h2"].items() for o in h["overlap"]],
              "trend.csv": [{"domain": d, **o} for d, h in results["h2"].items() for o in h["trend"]["pairs"]],
              "e1.csv": [{"domain": d, **r} for d, rows in results["e1"].items() for r in rows]}
    for name, rows in tables.items():
        write_csv(out / name, rows)
    return ["summary.md", *(n for n, rows in tables.items() if rows), *figures(results, out)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    print(", ".join(write_report(results, Path(args.out))))


if __name__ == "__main__":
    main()
