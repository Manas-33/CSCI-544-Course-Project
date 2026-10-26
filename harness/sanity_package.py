"""Build the slide-ready sanity package from a saved AR sanity evaluation. No models run here.

    python -m harness.sanity_package \\
        --run-dir results/bf16/sdpa/ar-sanity-bf16-sdpa-20261026-000459 \\
        --analysis-dir results/bf16/sdpa/ar-sanity-analysis-bf16-sdpa-20261026-000459 \\
        --previous-analysis results/bf16/ar-sanity-analysis-bf16-20261020-015535 \\
        --output-dir results/ppt_ar_sanity

Every number comes from the analysis metrics.json written by harness.evaluate, which is checked
against the run manifest. Files are UTF-8 with LF line endings.
"""

import argparse
import csv
import json
from pathlib import Path

from .run import digest

REQUIRED = 0.90
COLORS = ("#2563eb", "#f97316", "#16a34a", "#9333ea")


def load(run_dir, analysis_dir):
    manifest_raw = (Path(run_dir) / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    metrics = json.loads((Path(analysis_dir) / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("input_kind") != "sanity" or metrics.get("status") != "completed":
        raise ValueError("expected a completed sanity evaluation")
    if metrics["run_manifest_sha256"] != digest(manifest_raw):
        raise ValueError("analysis was not made from this run's manifest")
    if any("auroc_by_domain" not in m for m in metrics["models"].values()):
        raise ValueError("re-run harness.evaluate: metrics.json has no per-domain AUROC")
    return manifest, metrics


def rows_from(manifest, metrics):
    rows = []
    for key in sorted(metrics["models"]):
        m = metrics["models"][key]
        rows.append({
            "model": m["model_id"].split("/")[-1], "model_id": m["model_id"], "model_revision": m["model_revision"],
            "overall_auroc": round(m["auroc"], 4),
            "math_auroc": round(m["auroc_by_domain"]["math"], 4), "code_auroc": round(m["auroc_by_domain"]["code"], 4),
            "sanity_gate": "PASS" if m["auroc"] >= REQUIRED else "FAIL", "required_auroc": REQUIRED,
            "median_yes_no_mass": round(m["median_yes_no_mass"], 4),
            "peak_gpu_gib": round(m["peak_gpu_allocated_bytes"] / 2 ** 30, 2),
            "warm_mean_forward_seconds": round(m["warm_mean_forward_seconds"], 4),
            "precision": "BF16" if m["compute_dtype"] == "torch.bfloat16" else m["compute_dtype"],
            "quantization": m["quantization_mode"], "attention": manifest["attention_implementation"].upper(),
        })
    return rows


def figures(rows, counts, gpu, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "sanity", "font.size": 12})
    subtitle = (f"T1, {counts['candidates']} items ({counts['correct']} right, {counts['incorrect']} wrong; "
                f"26 math, 24 code) | {rows[0]['precision']} + {rows[0]['attention']} | {gpu}")

    def save(fig, name):
        fig.savefig(out / f"{name}.svg", metadata={"Date": None})
        fig.savefig(out / f"{name}.png", dpi=200, metadata={"Software": None})
        plt.close(fig)

    # AUROC by domain, grouped by model.
    fig, ax = plt.subplots(figsize=(10, 5.6))
    groups, width = ("overall", "math", "code"), 0.8 / len(rows)
    for i, row in enumerate(rows):
        xs = [g + (i - (len(rows) - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(xs, [row[f"{g}_auroc"] for g in groups], width * 0.92, color=COLORS[i], label=row["model"])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=11, fontweight="bold")
    # The bar applies to overall AUROC only, so the line is drawn over that group alone.
    ax.hlines(REQUIRED, -0.45, 0.45, color="#dc2626", linestyle="--", linewidth=2, label=f"Required overall AUROC ({REQUIRED:.2f})")
    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1, label="Chance (0.50)")
    ax.set(xticks=range(len(groups)), xticklabels=["Overall", "Math", "Code"], ylim=(0, 1.1), ylabel="AUROC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.suptitle("AR verifier sanity check: AUROC by domain", fontsize=16, fontweight="bold")
    ax.set_title(subtitle, fontsize=10, color="#4b5563")
    fig.tight_layout()
    save(fig, "auroc_by_domain")

    # Gate status: overall AUROC against the required bar.
    fig, ax = plt.subplots(figsize=(9, 2.2 + 0.7 * len(rows)))
    names = [r["model"] for r in rows][::-1]
    values = [r["overall_auroc"] for r in rows][::-1]
    bars = ax.barh(names, values, color=COLORS[:len(rows)][::-1], height=0.55)
    for bar, row in zip(bars, rows[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{row['overall_auroc']:.3f}  {row['sanity_gate']}", va="center", fontweight="bold",
                color="#15803d" if row["sanity_gate"] == "PASS" else "#b91c1c")
    ax.axvline(REQUIRED, color="#dc2626", linestyle="--", linewidth=2)
    ax.text(REQUIRED, len(rows) - 0.45, f"required ≥ {REQUIRED:.2f}", color="#dc2626", ha="center", fontsize=10)
    ax.set(xlim=(0, 1.15), xlabel="Overall AUROC")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Sanity gate: every verifier needs overall AUROC ≥ 0.90", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "sanity_gate_status")

    # Results table.
    header = ["Model", "Overall", "Math", "Code", "Gate"]
    cells = [[r["model"], f"{r['overall_auroc']:.3f}", f"{r['math_auroc']:.3f}", f"{r['code_auroc']:.3f}", r["sanity_gate"]]
             for r in rows]
    fig, ax = plt.subplots(figsize=(9, 1.1 + 0.55 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center", colWidths=[0.4, 0.14, 0.14, 0.14, 0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f2937")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif c == 4:
            cell.get_text().set_color("#15803d" if cells[r - 1][4] == "PASS" else "#b91c1c")
            cell.get_text().set_fontweight("bold")
    ax.set_title(f"AR sanity-check AUROC ({rows[0]['precision']} + {rows[0]['attention']})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "ar_sanity_table")


def summary_md(rows, counts, gpu, run_dir, analysis_dir, previous):
    lines = ["# AR verifier sanity check (slide package)", "",
             f"Generated by `python -m harness.sanity_package` from `{analysis_dir}` (run `{run_dir}`).", "",
             "## Result", "",
             "| Model | Overall AUROC | Math AUROC | Code AUROC | Gate (≥ 0.90) |", "|---|---:|---:|---:|---|"]
    lines += [f"| {r['model']} | {r['overall_auroc']:.3f} | {r['math_auroc']:.3f} | {r['code_auroc']:.3f} | {r['sanity_gate']} |"
              for r in rows]
    lines += ["", "## Setup", "",
              f"- The official 50-item sanity fixture (`bank/sanity/sanity_set.jsonl`): {counts['correct']} right and "
              f"{counts['incorrect']} wrong answers, 26 math and 24 code. Template T1, one forward pass, "
              "score = log P(Yes) − log P(No).",
              f"- {rows[0]['precision']} weights and compute, no quantization, {rows[0]['attention']} attention, {gpu}, batch size 1.",
              "- Full coverage: every model scored all 50 items, none failed or missing.",
              "- Cost: " + "; ".join(f"{r['model']} {r['peak_gpu_gib']} GiB peak, {r['warm_mean_forward_seconds']:.3f} s per item"
                                     for r in rows) + "."]
    if previous:
        lines += ["", "## What changed from the first run", "",
                  "The first BF16 run used eager attention. Its Qwen scores were wrong (it rejected answers like "
                  "\"8 × 5 = 40\" with high confidence), and switching to SDPA fixed them. Same models, revisions and prompts:", ""]
        lines += [f"- {r['model']}: overall {previous[r['model_id']]:.3f} (eager) → {r['overall_auroc']:.3f} (SDPA)"
                  for r in rows if r["model_id"] in previous]
        lines += ["", "The eager-attention run must not be used as evidence."]
    lines += ["", "## Reading it", "",
              "- PLAN.md §4 test 1: every verifier must reach overall AUROC ≥ 0.90 on this fixture before its scores are used. "
              "A model that still fails once its prompt is confirmed against the official chat template counts as a weak "
              "verifier (a gate A failure), not an open bug.",
              "- This is a scoring check, not research evidence. 25 right and 25 wrong items give wide intervals, "
              "so do not rank models by these numbers."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--analysis-dir", required=True)
    p.add_argument("--previous-analysis", help="earlier analysis of the same fixture, for the before/after line")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)
    manifest, metrics = load(args.run_dir, args.analysis_dir)
    rows = rows_from(manifest, metrics)
    previous = None
    if args.previous_analysis:
        old = json.loads((Path(args.previous_analysis) / "metrics.json").read_text(encoding="utf-8"))
        previous = {m["model_id"]: m["auroc"] for m in old["models"].values()}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gpu = metrics["gpu"]["name"]
    with (out / "ar_sanity_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    package = {"title": "AR verifier sanity check", "fixture": "bank/sanity/sanity_set.jsonl", "counts": metrics["counts"],
               "required_overall_auroc": REQUIRED, "run_dir": args.run_dir, "analysis_dir": args.analysis_dir,
               "run_manifest_sha256": metrics["run_manifest_sha256"], "gpu": gpu, "models": rows,
               "research_evidence": False}
    (out / "ar_sanity_results.json").write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "PPT_RESULTS_SUMMARY.md").write_text(
        summary_md(rows, metrics["counts"], gpu, args.run_dir, args.analysis_dir, previous), encoding="utf-8")
    figures(rows, metrics["counts"], gpu, out)
    print(f"Wrote the sanity package for {len(rows)} models to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
