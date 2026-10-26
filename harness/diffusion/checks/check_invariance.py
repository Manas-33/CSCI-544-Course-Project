"""Scoring-code test 3 (PLAN.md §4): repeat-run and batching invariance.

  * repeat runs: scoring the items twice, unbatched, gives Spearman > 0.999.
    ``--save`` keeps the scores; ``--previous`` compares with a saved run, e.g.
    from a fresh Colab runtime, which is the stronger repeat check;
  * batched vs unbatched: |delta s| < 0.05 on every item. Skipped for iLLaDA,
    which is always scored unbatched, so for it only repeat runs are checked.

Use items of different lengths (the sanity set or the development bank), so
batching actually adds padding.

    python -m harness.diffusion.checks.check_invariance --model dream --n-slots 8 \\
        --batch-size 8 --save /content/drive/MyDrive/cs544/dream_n8_run1.json
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import finish, load_items, message, model_args, scorer_from_args, spearman, verdicts_for

SPEARMAN_MIN = 0.999
BATCH_TOL = 0.05


def main(argv=None):
    ap = model_args(__doc__.splitlines()[0])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save", default=None, help="save this run's unbatched scores (JSON)")
    ap.add_argument("--previous", default=None, help="compare with scores saved by --save")
    args = ap.parse_args(argv)

    items = load_items(args.items)
    ids = [it["candidate_id"] for it in items]
    scorer = scorer_from_args(args)
    prompts = [scorer.prompt_ids(message(it)) for it in items]
    first = [v.score for v in verdicts_for(scorer, prompts, 1)]
    second = [v.score for v in verdicts_for(scorer, prompts, 1)]
    report = {"model": args.model, "revision": scorer.revision, "n_slots": args.n_slots,
              "answer_prefix": args.answer_prefix,
              "n_items": len(items), "spearman_min": SPEARMAN_MIN, "batch_tol": BATCH_TOL}
    checks = {}

    report["repeat_spearman"] = spearman(first, second)
    report["repeat_max_abs_diff"] = max(abs(a - b) for a, b in zip(first, second))
    checks["repeat"] = report["repeat_spearman"] > SPEARMAN_MIN
    print(f"repeat run: Spearman {report['repeat_spearman']:.6f}, "
          f"max |ds| {report['repeat_max_abs_diff']:.2e}")

    if args.previous:
        saved = json.loads(Path(args.previous).read_text())
        if ((saved.get("model"), saved.get("revision"), saved.get("n_slots"), saved.get("answer_prefix", "none"))
                != (args.model, scorer.revision, args.n_slots, args.answer_prefix)):
            raise SystemExit(f"{args.previous} was scored with other settings; compare like with like")
        shared = [i for i, cid in enumerate(ids) if cid in saved["scores"]]
        report["previous_spearman"] = spearman(
            [first[i] for i in shared], [saved["scores"][ids[i]] for i in shared])
        checks["previous"] = report["previous_spearman"] > SPEARMAN_MIN
        print(f"vs {args.previous}: Spearman {report['previous_spearman']:.6f} on {len(shared)} items")

    if scorer.spec.batched:
        if len({len(p) for p in prompts}) == 1:
            print("warning: all prompts have the same length, so batching adds no padding")
        batched = [v.score for v in verdicts_for(scorer, prompts, args.batch_size)]
        diffs = [abs(a - b) for a, b in zip(first, batched)]
        report["batch_size"] = args.batch_size
        report["batch_max_abs_diff"] = max(diffs)
        report["batch_worst_item"] = ids[diffs.index(max(diffs))]
        checks["batched"] = max(diffs) < BATCH_TOL
        print(f"batched (size {args.batch_size}) vs unbatched: max |ds| {max(diffs):.4f} "
              f"({report['batch_worst_item']})")
    else:
        print("batched check skipped: this model is scored unbatched")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(
            {"model": args.model, "revision": scorer.revision, "n_slots": args.n_slots,
             "answer_prefix": args.answer_prefix, "scores": dict(zip(ids, first))}, indent=2))

    report["checks"] = checks
    finish("test 3 invariance", "PASS" if all(checks.values()) else "FAIL", report, args.report)


if __name__ == "__main__":
    main()
