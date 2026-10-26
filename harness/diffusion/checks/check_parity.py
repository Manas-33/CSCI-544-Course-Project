"""Scoring-code test 2 (PLAN.md §4): parity with the official forward pass.

On up to 20 prompts, unbatched, in the model's official environment:

  * prompt: the scorer's token ids equal the ids built the official way
    (LLaDA: generate.py main(); iLLaDA: eval_illada.py generate_until();
    Dream: the README's apply_chat_template(..., return_dict=True));
  * first answer slot: the official sampler is run on prompt + N mask slots and
    stopped right after its first forward pass (LLaDA and iLLaDA: generate()
    from github.com/ML-GSAI/LLaDA; Dream: model.diffusion_generate(), which has
    already applied its one-position shift). Its distribution at the first slot
    must have the same top token as the scorer's, and a Yes/No score within
    1e-3. For Dream this is also the shift check.

The report saves the rendered prompt of the first item, for the prompt check
in PLAN.md §9.

    git clone https://github.com/ML-GSAI/LLaDA /content/LLaDA   # LLaDA and iLLaDA only
    python -m harness.diffusion.checks.check_parity --model llada --n-slots 8 \\
        --llada-repo /content/LLaDA
"""
from __future__ import annotations

import sys

import torch

from ..scorer import ANSWER_PREFIXES, verdict_from_logits
from .common import finish, load_items, message, model_args, scorer_from_args

N_PROMPTS = 20
SCORE_TOL = 1e-3
# Commit of github.com/ML-GSAI/LLaDA checked on 2026-10-19 (adds iLLaDA support).
LLADA_REPO_COMMIT = "9182493720ed723ef8031210d85959364e51cbe0"


class _FirstForward(Exception):
    def __init__(self, logits):
        super().__init__("stopped after the first forward pass")
        self.logits = logits


def official_prompt_ids(key, tokenizer, user_message, answer_prefix="none") -> list:
    """The prompt ids the official code builds, with the same answer prefix appended to the text."""
    messages = [{"role": "user", "content": user_message}]
    if key == "dream":
        if answer_prefix != "none":
            raise NotImplementedError("answer prefixes are not wired into Dream's official encoding")
        enc = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
        return enc["input_ids"][0].tolist()
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    text += ANSWER_PREFIXES[answer_prefix]
    if key == "llada":
        return tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokenizer(text)["input_ids"]  # iLLaDA instruct eval: add_bos_token=False


def official_slot_logits(scorer, prompt_ids, llada_generate=None) -> torch.Tensor:
    """Logits the official sampler uses for the first answer slot."""
    n = scorer.n_slots
    prompt = torch.tensor([prompt_ids], device=scorer.device)
    first_slot = len(prompt_ids)
    if scorer.spec.key == "dream":
        def hook(step, x, logits):  # called after the official shift
            raise _FirstForward(logits)
        try:
            scorer.model.diffusion_generate(
                prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=n, steps=n,
                temperature=0.0, alg="entropy", return_dict_in_generate=True,
                generation_logits_hook_func=hook)
        except _FirstForward as stop:
            return stop.logits[0, first_slot]
    else:
        def hook(module, inputs, output):
            raise _FirstForward(output.logits)
        handle = scorer.model.register_forward_hook(hook)
        try:
            llada_generate(scorer.model, prompt, steps=n, gen_length=n, block_length=n,
                           temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
                           mask_id=scorer.spec.mask_id)
        except _FirstForward as stop:
            return stop.logits[0, first_slot]
        finally:
            handle.remove()
    raise RuntimeError("the official sampler returned without a forward pass")


def balanced_items(items: list, n: int) -> list:
    """The first n items, half from each domain when items carry one (the sanity set
    lists its code items first, so a plain slice would test no math prompt)."""
    domains = sorted({it.get("domain") for it in items})
    if len(domains) < 2:
        return items[:n]
    per = n // len(domains)
    chosen = [it for d in domains for it in [x for x in items if x.get("domain") == d][:per]]
    rest = [it for it in items if it not in chosen]
    return chosen + rest[:n - len(chosen)]


def main(argv=None):
    ap = model_args(__doc__.splitlines()[0])
    ap.add_argument("--llada-repo", default=None,
                    help=f"clone of github.com/ML-GSAI/LLaDA (commit {LLADA_REPO_COMMIT[:7]}); "
                         "needed for llada and illada")
    ap.add_argument("--n", type=int, default=N_PROMPTS)
    args = ap.parse_args(argv)

    llada_generate = None
    if args.model != "dream":
        if not args.llada_repo:
            ap.error("--llada-repo is required for llada and illada")
        sys.path.insert(0, args.llada_repo)
        from generate import generate as llada_generate  # the official sampler

    items = balanced_items(load_items(args.items), args.n)
    scorer = scorer_from_args(args)
    results, failures = [], 0
    for it in items:
        user_message = message(it)
        ours_ids = scorer.prompt_ids(user_message)
        official_ids = official_prompt_ids(args.model, scorer.tokenizer, user_message, args.answer_prefix)
        ours = scorer.score_ids([ours_ids])[0]
        ref = verdict_from_logits(
            official_slot_logits(scorer, official_ids, llada_generate), scorer.tokens)
        row = {
            "candidate_id": it["candidate_id"],
            "same_prompt_ids": ours_ids == official_ids,
            "top_token": [ours.top_id, ref.top_id],
            "score": [ours.score, ref.score],
            "score_diff": abs(ours.score - ref.score),
        }
        row["ok"] = (row["same_prompt_ids"] and ours.top_id == ref.top_id
                     and row["score_diff"] <= SCORE_TOL)
        failures += not row["ok"]
        results.append(row)
        print(f"{it['candidate_id']:<14} prompt {'same' if row['same_prompt_ids'] else 'DIFFERENT'}"
              f"  top {ours.top_id}/{ref.top_id}  |ds| {row['score_diff']:.2e}"
              f"  {'ok' if row['ok'] else 'MISMATCH'}")

    example = message(items[0])
    if failures:
        status = "FAIL"
    elif len(items) < N_PROMPTS:
        status = f"INCOMPLETE (ran {len(items)} prompts, need {N_PROMPTS})"
    else:
        status = "PASS"
    finish("test 2 parity", status, {
        "model": args.model, "revision": scorer.revision, "n_slots": args.n_slots,
        "answer_prefix": args.answer_prefix,
        "llada_repo_commit": LLADA_REPO_COMMIT if args.model != "dream" else None,
        "score_tol": SCORE_TOL, "n_prompts": len(items), "failures": failures,
        "rendered_prompt_example": scorer.render(example) + ANSWER_PREFIXES[scorer.answer_prefix],
        "prompt_ids_example": scorer.prompt_ids(example),
        "results": results,
    }, args.report)


if __name__ == "__main__":
    main()
