"""Scoring-code test 1 (PLAN.md §4): sanity-set AUROC >= 0.90.

Scores the hand-written sanity set (25 obviously right and 25 obviously wrong
candidates, outside the bank; label 1 = correct) with T1. A lower AUROC almost
always means a bug: wrong mask id, missing shift, wrong template, padding leak.
Also prints the median Yes+No mass (the usable-scores gate needs >= 0.5) and the
most likely token at the first answer slot, which shows what the model wants to
say when the mass is low.

    python -m harness.diffusion.checks.check_sanity --model llada --n-slots 8
"""
from __future__ import annotations

import statistics

from ..interface import score_row, write_jsonl
from .common import auroc, finish, load_items, message, model_args, scorer_from_args, verdicts_for

AUROC_MIN = 0.90
PER_LABEL = 25


def main(argv=None):
    ap = model_args(__doc__.splitlines()[0])
    ap.add_argument("--batch-size", type=int, default=8, help="ignored for iLLaDA")
    ap.add_argument("--out", default=None, help="also save the score rows here")
    args = ap.parse_args(argv)

    items = load_items(args.items)
    labels = [int(it["label"]) for it in items]
    scorer = scorer_from_args(args)
    prompts = [scorer.prompt_ids(message(it)) for it in items]
    verdicts = verdicts_for(scorer, prompts, args.batch_size if scorer.spec.batched else 1)

    print(f"{'candidate_id':<14} label  {'score':>8}  {'yes+no':>6}  top token")
    for it, y, v in zip(items, labels, verdicts):
        top = scorer.tokenizer.decode([v.top_id])
        print(f"{it['candidate_id']:<14} {y:>5}  {v.score:>8.3f}  {v.mass:>6.3f}  {top!r}")

    scores = [v.score for v in verdicts]
    value = auroc(scores, labels)
    median_mass = statistics.median(v.mass for v in verdicts)
    n_pos, n_neg = labels.count(1), labels.count(0)
    print(f"\nAUROC {value:.3f} on {n_pos} correct + {n_neg} wrong; median Yes+No mass {median_mass:.3f}")
    print(f"Yes set {scorer.tokens.yes_ids}, No set {scorer.tokens.no_ids}, dropped {scorer.tokens.dropped}")

    if args.out:
        write_jsonl(args.out, [
            score_row(bank_hash="sanity", candidate_id=it["candidate_id"], model=args.model,
                      revision=scorer.revision, template="T1", n_slots=args.n_slots, verdict=v)
            for it, v in zip(items, verdicts)])

    if n_pos < PER_LABEL or n_neg < PER_LABEL:
        status = f"INCOMPLETE (sanity set needs {PER_LABEL} correct + {PER_LABEL} wrong)"
    else:
        status = "PASS" if value >= AUROC_MIN else "FAIL"
    finish("test 1 sanity", status, {
        "model": args.model, "revision": scorer.revision, "n_slots": args.n_slots,
        "auroc": value, "auroc_min": AUROC_MIN, "n_correct": n_pos, "n_wrong": n_neg,
        "median_yes_no_mass": median_mass,
        "items": [{"candidate_id": it["candidate_id"], "label": y, "score": v.score,
                   "yes_no_mass": v.mass, "top_token": scorer.tokenizer.decode([v.top_id])}
                  for it, y, v in zip(items, labels, verdicts)],
    }, args.report)


if __name__ == "__main__":
    main()
