"""Reconstruction score against the official code, on real weights (PLAN.md §4).

On a few items, in the model's official environment:

  * the prompt ids equal the official chat prompt of the problem, and no token spans the
    prompt/answer boundary;
  * LLaDA: the official get_log_likelihood() (github.com/ML-GSAI/LLaDA) runs one group of 16
    samples. The noisy inputs it feeds the model are captured and scored by our code, which must
    give the same estimate within 1e-3 per token when the official cross-entropy is given float32
    logits, as ours is (the official one computes it in BF16; that value is reported beside it).
    The official mask counts must follow ours and the prompt must never be masked;
  * Dream has no official log-likelihood. The official sampler's first forward pass, which has
    already applied its one-position shift, is run on the same noisy input: the masked tokens'
    log-probabilities must agree within 1e-3 per token. The value without the shift is reported;
  * the same seed gives the identical estimate twice (unbatched).

It also reports how far 16 samples per forward pass move the estimate from one per pass.

    python -m harness.diffusion.checks.check_reconstruction --model llada --llada-repo /content/LLaDA
"""
from __future__ import annotations

import argparse
import math
import sys

import torch

from ..reconstruction import (MC_GROUP, MODELS, draw_masks, encode_pair, item_generator, mask_counts, reconstruct,
                              sample_logliks)
from ..scorer import load_scorer
from .check_parity import LLADA_REPO_COMMIT, _FirstForward, balanced_items, official_prompt_ids
from .common import SANITY_SET, finish, load_items

N_ITEMS = 6
TOL = 1e-3  # per answer token
SEED = 544


def official_llada(scorer, pair, get_log_likelihood, upcast):
    """(official estimate, the noisy inputs it fed the model) for one group of samples."""
    captured = []

    def hook(module, args, kwargs, output):
        captured.append((args[0] if args else kwargs["input_ids"]).detach().cpu().clone())
        if upcast:
            output.logits = output.logits.float()
        return output

    handle = scorer.model.register_forward_hook(hook, with_kwargs=True)
    try:
        torch.manual_seed(SEED)
        value = get_log_likelihood(
            scorer.model, torch.tensor(pair.prompt, device=scorer.device),
            torch.tensor(pair.answer, device=scorer.device), mc_num=MC_GROUP, batch_size=MC_GROUP,
            cfg_scale=0.0, mask_id=scorer.spec.mask_id)
    finally:
        handle.remove()
    return float(value), torch.cat(captured)


def check_llada(scorer, pair, get_log_likelihood) -> dict:
    n_prompt, n_answer = len(pair.prompt), len(pair.answer)
    official, noisy = official_llada(scorer, pair, get_log_likelihood, upcast=True)
    official_bf16, _ = official_llada(scorer, pair, get_log_likelihood, upcast=False)
    masks = noisy[:, n_prompt:] == scorer.spec.mask_id
    counts = masks.sum(dim=1)
    values = sample_logliks(scorer, pair.prompt, pair.answer, masks, micro_batch=MC_GROUP)
    ours = math.fsum(values) / len(values)
    return {
        "official_per_token": official / n_answer, "ours_per_token": ours / n_answer,
        "diff_per_token": abs(ours - official) / n_answer,
        "official_bf16_cross_entropy_per_token": official_bf16 / n_answer,
        "prompt_never_masked": bool((noisy[:, :n_prompt] == torch.tensor(pair.prompt)).all()),
        "same_mask_counts": counts.tolist() == mask_counts(int(counts[0]), n_answer, MC_GROUP).tolist(),
    }


@torch.no_grad()
def check_dream(scorer, pair, candidate_id) -> dict:
    n_prompt, n_answer = len(pair.prompt), len(pair.answer)
    # The sampler appends masks to its prompt, so the noisy answer must end with one.
    mask = draw_masks(n_answer, MC_GROUP, item_generator(SEED, candidate_id))[0].clone()
    mask[-1] = True
    sequence = torch.tensor(pair.prompt + pair.answer)
    noisy = sequence.clone()
    noisy[n_prompt:][mask] = scorer.spec.mask_id
    seen = {}

    def hook(step, x, logits):  # called after the official shift
        seen["input"] = x.detach().cpu()
        raise _FirstForward(logits)

    prompt = noisy[:-1][None].to(scorer.device)
    try:
        scorer.model.diffusion_generate(
            prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=1, steps=1, temperature=0.0,
            alg="entropy", return_dict_in_generate=True, generation_logits_hook_func=hook)
        raise RuntimeError("the official sampler returned without a forward pass")
    except _FirstForward as stop:
        logits = stop.logits[0]
    where = n_prompt + mask.nonzero(as_tuple=True)[0]

    def total(rows):
        log_probs = torch.log_softmax(rows.float(), dim=-1).cpu()
        return float(log_probs.gather(1, sequence[where][:, None]).double().sum())

    k = len(where)
    official = total(logits[where.to(logits.device)])
    ours = sample_logliks(scorer, pair.prompt, pair.answer, mask[None])[0] * k / n_answer
    batch = noisy[None].to(scorer.device)
    unshifted = scorer.forward(batch, torch.ones_like(batch))[0]
    return {"official_per_token": official / k, "ours_per_token": ours / k, "diff_per_token": abs(ours - official) / k,
            "without_shift_per_token": total(unshifted[where.to(unshifted.device)]) / k, "masked_tokens": k,
            "same_noisy_input": bool((seen["input"][0] == noisy).all())}


@torch.no_grad()
def check_item(scorer, item, get_log_likelihood=None) -> dict:
    cid = item["candidate_id"]
    pair = encode_pair(scorer.tokenizer, item["problem"], item["candidate"])
    row = {"candidate_id": cid, "n_prompt_tokens": len(pair.prompt), "n_tokens": len(pair.answer),
           "boundary": pair.boundary,
           "same_prompt_ids": list(pair.prompt) == official_prompt_ids(scorer.spec.key, scorer.tokenizer, item["problem"])}
    if scorer.spec.key == "dream":
        row.update(check_dream(scorer, pair, cid))
        ok = row["same_noisy_input"]
    else:
        row.update(check_llada(scorer, pair, get_log_likelihood))
        ok = row["prompt_never_masked"] and row["same_mask_counts"]
    first, second = (reconstruct(scorer, pair, cid, SEED, MC_GROUP, 1) for _ in range(2))
    batched = reconstruct(scorer, pair, cid, SEED, MC_GROUP, MC_GROUP)
    row["repeat_identical"] = first == second
    row["micro_batch_16_vs_1_per_token"] = abs(batched.score - first.score)
    row["ok"] = bool(ok and row["same_prompt_ids"] and row["boundary"] == "clean" and row["repeat_identical"]
                     and row["diff_per_token"] <= TOL)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--llada-repo", default=None,
                    help=f"clone of github.com/ML-GSAI/LLaDA (commit {LLADA_REPO_COMMIT[:7]}); needed for llada")
    ap.add_argument("--items", default=str(SANITY_SET), help="JSONL with candidate_id, problem, candidate")
    ap.add_argument("--n", type=int, default=N_ITEMS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    ap.add_argument("--report", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    get_log_likelihood = None
    if args.model == "llada":
        if not args.llada_repo:
            ap.error("--llada-repo is required for llada")
        sys.path.insert(0, args.llada_repo)
        from get_log_likelihood import get_log_likelihood  # the official estimate

    items = balanced_items(load_items(args.items), args.n)
    scorer = load_scorer(args.model, 1, device=args.device, strict_env=not args.allow_env_mismatch)
    rows = []
    for item in items:
        rows.append(check_item(scorer, item, get_log_likelihood))
        r = rows[-1]
        print(f"{r['candidate_id']:<14} {r['n_tokens']:>4} tokens  ours {r['ours_per_token']:.4f}  official "
              f"{r['official_per_token']:.4f}  |d| {r['diff_per_token']:.1e}  16-vs-1 "
              f"{r['micro_batch_16_vs_1_per_token']:.1e}  {'ok' if r['ok'] else 'MISMATCH'}")
    failures = sum(not r["ok"] for r in rows)
    status = "FAIL" if failures else ("PASS" if len(rows) >= N_ITEMS else f"INCOMPLETE ({len(rows)} items)")
    finish("reconstruction parity", status, {
        "model": args.model, "revision": scorer.revision, "tol_per_token": TOL, "failures": failures,
        "llada_repo_commit": LLADA_REPO_COMMIT if args.model == "llada" else None,
        "micro_batch_16_vs_1_max_per_token": max(r["micro_batch_16_vs_1_per_token"] for r in rows),
        "results": rows,
    }, args.report)


if __name__ == "__main__":
    main()
