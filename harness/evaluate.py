"""Offline, exploratory development-bank evaluation. Labels enter only here."""

import argparse
from collections import defaultdict
import csv
from itertools import groupby
import math
from pathlib import Path
import statistics
import sys
import textwrap

from .data import BANK_MAPPING, load_records, parse_json, read_jsonl
from .prompts import TEMPLATES, messages
from .run import digest, finite_number, json_bytes, validate_config


def index_rows(rows, key, source):
    indexed = {}
    for number, row in enumerate(rows, 1):
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{source}, line {number}: missing/nonstring {key}")
        if identifier in indexed:
            raise ValueError(f"{source}, line {number}: duplicate ID {identifier!r}")
        indexed[identifier] = row
    return indexed


def require_same_ids(expected, actual, source):
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise ValueError(f"{source} ID mismatch: missing={sorted(missing)}, unknown={sorted(extra)}")


def join_labels(records, label_rows, bank_hash):
    labels = index_rows(label_rows, "candidate_id", "labels")
    require_same_ids({r["id"] for r in records}, labels, "labels")
    for record in records:
        row = labels[record["id"]]
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError(f"unknown label for {record['id']!r}: expected integer 0 or 1")
        if "bank_sha256" in row and row["bank_sha256"] != bank_hash:
            raise ValueError(f"label bank hash mismatch for {record['id']!r}")
        for key in ("problem_id", "pair_id"):
            if key in row and row[key] != record[key]:
                raise ValueError(f"label {key} mismatch for {record['id']!r}")
    return {candidate_id: row["label"] for candidate_id, row in labels.items()}


def matched_pairs(records, labels):
    groups = defaultdict(list)
    for record in records:
        groups[record["pair_id"]].append(record)
    pairs = []
    for pair_id, members in groups.items():
        if (len(members) != 2 or len({r["problem_id"] for r in members}) != 1
                or {labels[r["id"]] for r in members} != {0, 1}):
            raise ValueError(f"malformed pair {pair_id!r}: require one correct and one wrong candidate")
        pairs.append(tuple(next(r["id"] for r in members if labels[r["id"]] == label) for label in (1, 0)))
    return pairs


def roc_curve(labels, scores):
    """Tie-aware empirical ROC/AUROC; higher scores mean more likely correct."""
    if len(labels) != len(scores) or not scores:
        raise ValueError("ROC requires equally sized nonempty label/score arrays")
    if any(type(y) is not int or y not in (0, 1) for y in labels) or not all(map(finite_number, scores)):
        raise ValueError("ROC requires binary labels and finite scores")
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    tp = fp = 0
    xs, ys = [0.0], [0.0]
    for _, group in groupby(sorted(zip(scores, labels), reverse=True), key=lambda x: x[0]):
        outcomes = [y for _, y in group]
        tp += sum(outcomes)
        fp += len(outcomes) - sum(outcomes)
        xs.append(fp / negatives)
        ys.append(tp / positives)
    area = math.fsum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2 for i in range(1, len(xs)))
    return xs, ys, area


def pair_ranking(pairs, scores):
    if not pairs:
        raise ValueError("within-pair ranking requires matched pairs")
    return math.fsum(1.0 if scores[good] > scores[bad] else 0.5 if scores[good] == scores[bad] else 0.0
                     for good, bad in pairs) / len(pairs)


def confusion(labels, accepted):
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for label, accept in zip(labels, accepted):
        counts[("tp" if accept else "fn") if label else ("fp" if accept else "tn")] += 1
    return {**counts, "accuracy": (counts["tp"] + counts["tn"]) / len(labels),
            "tar": counts["tp"] / (counts["tp"] + counts["fn"]),
            "far": counts["fp"] / (counts["fp"] + counts["tn"])}


def zero_diagnostics(labels, a_scores, b_scores):
    a, b = [s >= 0 for s in a_scores], [s >= 0 for s in b_scores]
    overlap = {key: 0 for key in ("both", "A_only", "B_only", "neither")}
    for label, accept_a, accept_b in zip(labels, a, b):
        if label == 0:
            key = "both" if accept_a and accept_b else "A_only" if accept_a else "B_only" if accept_b else "neither"
            overlap[key] += 1
    return {"description": "Exploratory fixed-zero diagnostics; not matched-TAR or validation-selected",
            "thresholds": {"A": 0, "B": 0}, "A": confusion(labels, a), "B": confusion(labels, b),
            "AND": confusion(labels, [x and y for x, y in zip(a, b)]), "wrong_accepted_by": overlap}


def score_coverage(records, scores, bank_hash, model):
    expected = {r["id"] for r in records}
    extra = set(scores) - expected
    if extra:
        raise ValueError(f"model {model}: unknown score IDs {sorted(extra)}")
    completed, failed, missing = [], [], []
    for record in records:
        cid = record["id"]
        row = scores.get(cid)
        if row is None:
            missing.append(cid)
            continue
        if row.get("candidate_id") != cid or row.get("bank_sha256") != bank_hash or row.get("model") != model:
            raise ValueError(f"model {model}, ID {cid!r}: incompatible score identity/bank hash")
        if any(row.get(key) != record[key] for key in ("problem_id", "pair_id")):
            raise ValueError(f"model {model}, ID {cid!r}: inconsistent grouping")
        status = row.get("status")
        if status in ("failed", "missing"):
            (failed if status == "failed" else missing).append(cid)
            continue
        if status != "completed":
            raise ValueError(f"model {model}, ID {cid!r}: unknown score status {status!r}")
        numeric = ("score", "p_yes", "p_no", "yes_no_mass", "forward_seconds")
        if not all(finite_number(row.get(key)) for key in numeric):
            raise ValueError(f"model {model}, ID {cid!r}: nonfinite/missing score measurements")
        if (any(not 0 <= row[key] <= 1.000001 for key in ("p_yes", "p_no", "yes_no_mass"))
                or not math.isclose(row["yes_no_mass"], row["p_yes"] + row["p_no"], abs_tol=1e-7, rel_tol=1e-6)
                or row["forward_seconds"] < 0 or type(row.get("cold")) is not bool
                or type(row.get("input_tokens")) is not int or not 1 <= row["input_tokens"] <= 1760):
            raise ValueError(f"model {model}, ID {cid!r}: invalid probabilities/timing/token count")
        completed.append(cid)
    return {"expected": len(records), "completed": len(completed), "failed": len(failed),
            "missing": len(missing), "failed_ids": failed, "missing_ids": missing}


def check_manifest(manifest, records, bank_hash):
    if manifest.get("score_contract_version") != 2 or manifest.get("input_sha256") != bank_hash:
        raise ValueError("incompatible scoring contract or bank SHA-256; score this exact bank with the integrated harness")
    config = manifest["effective_config"]
    validate_config(config)
    if config.get("input_format") != "dev_bank":
        raise ValueError("evaluation requires a dev_bank scoring run")
    template = config["template"]
    limit = manifest.get("limit")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("invalid manifest limit")
    selected = records[:limit]
    if (manifest.get("validated_records") != len(records) or manifest.get("selected_records") != len(selected)
            or manifest.get("effective_config_sha256") != digest(json_bytes(config))
            or manifest.get("template") != template or manifest.get("template_text") != TEMPLATES[template]
            or manifest.get("template_sha256") != digest(TEMPLATES[template].encode("utf-8"))
            or manifest.get("prompt_sha256") != digest(json_bytes([messages(r, template) for r in selected]))):
        raise ValueError("manifest input count, config, or prompt hash mismatch")


def review_rows(records, labels, scores, thresholds, threshold_source):
    for record in records:
        cid = record["id"]
        row = {"candidate_id": cid, "problem_id": record["problem_id"], "pair_id": record["pair_id"],
               "question": record["problem"], "candidate_solution": record["candidate_answer"], "label": labels[cid]}
        for model in ("A", "B"):
            row.update({f"{key}_{model}": scores[model][cid][key] for key in ("score", "p_yes", "p_no", "yes_no_mass")})
        if thresholds is not None:
            a, b = (scores[name][cid]["score"] >= thresholds[name] for name in ("A", "B"))
            if labels[cid] == 0:
                category = "shared_false_accept" if a and b else "A_only_false_accept" if a else "B_only_false_accept" if b else "both_correct_reject"
            else:
                category = "both_correct_accept" if a and b else "B_only_false_reject" if a else "A_only_false_reject" if b else "shared_false_reject"
            row.update(accept_A=a, accept_B=b, accept_AND=a and b, category=category,
                       threshold_A=thresholds["A"], threshold_B=thresholds["B"], threshold_source=threshold_source)
        yield row


def write_csv(path, rows):
    rows = iter(rows)
    first = next(rows)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def make_figure(path, labels, values, metrics, diagnostic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for name, color in (("A", "#2458a6"), ("B", "#b54c16")):
        xs, ys, area = roc_curve(labels, values[name])
        model = metrics["models"][name]["model_id"].split("/")[-1]
        axes[0].plot(xs, ys, lw=2, color=color, label=f"{name}: {model}\nAUROC = {area:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="0.65", lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02), title="Continuous-score discrimination")
    axes[0].legend(loc="lower right", fontsize=8)
    for label, color, marker, text in ((1, "#2458a6", "o", "Correct final answer"), (0, "#b54c16", "x", "Incorrect final answer")):
        positions = [i for i, y in enumerate(labels) if y == label]
        axes[1].scatter([values["A"][i] for i in positions], [values["B"][i] for i in positions],
                        c=color, marker=marker, s=30, alpha=0.65, label=text)
    if diagnostic:
        axes[1].axvline(0, color="0.4", linestyle="--", lw=1)
        axes[1].axhline(0, color="0.4", linestyle="--", lw=1)
    axes[1].set(xlabel="A score (higher = more likely correct)", ylabel="B score (higher = more likely correct)", title="Candidate scores" + ("; exploratory zero lines" if diagnostic else ""))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.15)
    counts = metrics["counts"]
    fig.suptitle(f"Development bank — exploratory\n{counts['candidates']} candidates · {counts['problems']} problems · {counts['matched_pairs']} matched pairs", fontsize=14)
    precision = " | ".join(f"{name}: {metrics['models'][name]['quantization_mode']} weights, {metrics['models'][name]['compute_dtype']} compute" for name in ("A", "B"))
    fig.text(0.5, 0.025, textwrap.fill(precision, 110), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.89))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def evaluate(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("evaluation output directory already exists")
    records, bank_hash = load_records(args.bank, BANK_MAPPING, "dev_bank")
    label_rows, label_hash = read_jsonl(args.labels)
    labels = join_labels(records, label_rows, bank_hash)
    pairs = matched_pairs(records, labels)
    run_dir = Path(args.run_dir)
    manifest_raw = (run_dir / "manifest.json").read_bytes()
    manifest = parse_json(manifest_raw.decode("utf-8"))
    check_manifest(manifest, records, bank_hash)
    scores, coverage = {}, {}
    for name in ("A", "B"):
        path = run_dir / f"scores_{name}.jsonl"
        rows = read_jsonl(path)[0] if path.exists() and path.stat().st_size else []
        scores[name] = index_rows(rows, "id", path.name)
        coverage[name] = score_coverage(records, scores[name], bank_hash, name)
    counts = {"candidates": len(records), "problems": len({r["problem_id"] for r in records}),
              "matched_pairs": len(pairs), "correct": sum(labels.values()), "incorrect": len(labels) - sum(labels.values())}
    metrics = {"status": "incomplete", "description": "Development bank — exploratory; initial AR+AR baseline",
               "bank_sha256": bank_hash, "labels_sha256": label_hash, "run_manifest_sha256": digest(manifest_raw),
               "counts": counts, "coverage": coverage, "run_kind": manifest.get("run_kind"), "gpu": manifest.get("gpu"),
               "interpretation": "Not test results, matched-TAR evidence, or proof of ensemble benefit. Final-answer labels do not establish correct reasoning; token-event probabilities are not calibrated correctness confidence."}
    if any(c["completed"] != len(records) for c in coverage.values()) or manifest.get("status") != "completed":
        output.mkdir(parents=True, exist_ok=False)
        (output / "metrics.json").write_bytes(json_bytes(metrics) + b"\n")
        print(json_bytes({"status": "incomplete", "coverage": coverage}).decode(), file=sys.stderr)
        return 1
    for name in ("A", "B"):
        path = run_dir / f"scores_{name}.jsonl"
        if manifest.get("outputs_sha256", {}).get(path.name) != digest(path.read_bytes()):
            raise ValueError(f"score checksum mismatch: {path.name}")
    ordered_labels = [labels[r["id"]] for r in records]
    values = {name: [scores[name][r["id"]]["score"] for r in records] for name in ("A", "B")}
    metrics.update(status="completed", models={})
    summary = []
    for name in ("A", "B"):
        meta = manifest["models"][name]
        rows = list(scores[name].values())
        warm = [row["forward_seconds"] for row in rows if not row["cold"]]
        model_metrics = {
            "model_id": meta["requested"]["model_id"], "model_revision": meta.get("resolved_model_revision"),
            "auroc": roc_curve(ordered_labels, values[name])[2],
            "within_pair_ranking_accuracy": pair_ranking(pairs, {cid: row["score"] for cid, row in scores[name].items()}),
            "warm_forward_seconds": math.fsum(warm), "warm_calls": len(warm),
            "warm_mean_forward_seconds": statistics.mean(warm) if warm else None,
            "cold_calls_excluded": sum(row["cold"] for row in rows),
            "median_yes_no_mass": statistics.median(row["yes_no_mass"] for row in rows),
            "loading_seconds": meta["loading_seconds"], "peak_gpu_allocated_bytes": meta["peak_gpu_allocated_bytes"],
            "actual_dtype": meta["actual_dtype"], "compute_dtype": manifest["compute_dtype"],
            "quantization_mode": meta.get("actual_quantization") or "unquantized", "quantization": meta["quantization"],
        }
        metrics["models"][name] = model_metrics
        summary.append({"model": name, **counts, **{k: v for k, v in model_metrics.items() if k != "quantization"}})
    thresholds = manifest["effective_config"]["thresholds"]
    threshold_source = "explicit_scoring_run_thresholds"
    if args.diagnostic_threshold_zero:
        diagnostics = zero_diagnostics(ordered_labels, values["A"], values["B"])
        metrics["diagnostics"] = diagnostics
        thresholds, threshold_source = diagnostics["thresholds"], "exploratory_fixed_zero"
        for row in summary:
            row.update({f"exploratory_zero_{key}": value for key, value in diagnostics[row["model"]].items()})
            row.update({f"exploratory_zero_AND_{key}": value for key, value in diagnostics["AND"].items()})
    metrics["review_thresholds"] = thresholds
    metrics["review_threshold_source"] = threshold_source if thresholds is not None else None
    # Check plotting availability only after coverage, without importing GPU libraries.
    import matplotlib  # noqa: F401
    output.mkdir(parents=True, exist_ok=False)
    make_figure(output / "development.png", ordered_labels, values, metrics, args.diagnostic_threshold_zero)
    (output / "metrics.json").write_bytes(json_bytes(metrics) + b"\n")
    write_csv(output / "summary.csv", summary)
    write_csv(output / "review.csv", review_rows(records, labels, scores, thresholds, threshold_source))
    print(f"Exploratory evaluation saved to {output}; {len(records)} candidates, {len(pairs)} matched pairs")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "labels", "run-dir", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--diagnostic-threshold-zero", action="store_true",
                        help="exploratory fixed-zero diagnostics; never fits or changes scoring thresholds")
    args = parser.parse_args(argv)
    try:
        return evaluate(args)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
