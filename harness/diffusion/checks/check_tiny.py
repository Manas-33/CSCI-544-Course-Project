"""CPU smoke test: the official model code with tiny random weights.

Run it in each model's environment before spending GPU time. It downloads only
the pinned config, model code and tokenizer (no weights), builds a 2-layer model
with random weights from the official code, and checks on the sanity-set prompts:

  * the mask id is the tokenizer's mask token; prints the Yes/No sets;
  * the scorer's prompt ids equal the ids built the official way, and the
    prompt does not start with a doubled BOS;
  * the scorer runs on the official model class (call signature, outputs);
  * parity with the official sampler's first forward pass, as in test 2;
  * batched and unbatched scores agree (LLaDA and Dream).

Random weights make the scores meaningless; only agreement is checked. Passing
this does not replace tests 1-3 on the real weights.

    python -m harness.diffusion.checks.check_tiny --model llada --llada-repo /content/LLaDA
"""
from __future__ import annotations

import sys

import torch

from ..scorer import ANSWER_PREFIXES, SCORERS, SPECS, check_environment, verdict_from_logits
from .check_parity import SCORE_TOL, official_prompt_ids, official_slot_logits
from .common import SANITY_SET, finish, load_items, message, verdicts_for

SHRINK = {
    "llada": dict(n_layers=2, d_model=64, n_heads=4, n_kv_heads=4, mlp_hidden_size=128,
                  init_device="cpu"),
    "illada": dict(num_hidden_layers=2, hidden_size=64, num_attention_heads=4,
                   num_key_value_heads=2, intermediate_size=128),
    "dream": dict(num_hidden_layers=2, hidden_size=64, num_attention_heads=4,
                  num_key_value_heads=2, intermediate_size=128, max_window_layers=2),
}


def tiny_model(spec, revision):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(spec.hf_id, revision=revision, trust_remote_code=True)
    for name, value in SHRINK[spec.key].items():
        setattr(config, name, value)
    model = AutoModel.from_config(config, trust_remote_code=True).float().eval()
    torch.manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.fill_(1.0)
            else:
                p.normal_(0.0, 0.1)
    return model


def main(argv=None):
    import argparse

    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(SPECS))
    ap.add_argument("--llada-repo", default=None, help="clone of github.com/ML-GSAI/LLaDA")
    ap.add_argument("--items", default=str(SANITY_SET))
    ap.add_argument("--revision", default=None)
    ap.add_argument("--answer-prefix", default="none", choices=sorted(ANSWER_PREFIXES))
    ap.add_argument("--allow-env-mismatch", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    spec = SPECS[args.model]
    if not args.allow_env_mismatch:
        check_environment(spec)
    llada_generate = None
    if args.model != "dream":
        if not args.llada_repo:
            ap.error("--llada-repo is required for llada and illada")
        sys.path.insert(0, args.llada_repo)
        from generate import generate as llada_generate

    revision = args.revision or spec.revision
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, revision=revision, trust_remote_code=True)
    model = tiny_model(spec, revision)
    items = load_items(args.items)
    problems = []

    mask_token = tokenizer.convert_ids_to_tokens(spec.mask_id)
    print(f"mask id {spec.mask_id} -> {mask_token!r}")
    if mask_token != spec.mask_token:
        problems.append(f"mask id {spec.mask_id} is {mask_token!r}, not {spec.mask_token!r}")

    report = {"model": args.model, "revision": revision, "mask_token": mask_token, "n_slots": {}}
    for n_slots in (1, 8, 32):
        scorer = SCORERS[args.model](model, tokenizer, spec, n_slots, device="cpu", revision=revision,
                                     answer_prefix=args.answer_prefix)
        prompts = []
        worst = 0.0
        for it in items:
            user_message = message(it)
            ids = scorer.prompt_ids(user_message)
            prompts.append(ids)
            if ids != official_prompt_ids(args.model, tokenizer, user_message, args.answer_prefix):
                problems.append(f"{it['candidate_id']}: prompt ids differ from the official ones")
            if len(ids) > 1 and ids[0] == ids[1] == tokenizer.bos_token_id:
                problems.append(f"{it['candidate_id']}: doubled BOS")
            ours = scorer.score_ids([ids])[0]
            ref = verdict_from_logits(official_slot_logits(scorer, ids, llada_generate), scorer.tokens)
            worst = max(worst, abs(ours.score - ref.score))
            if ours.top_id != ref.top_id or abs(ours.score - ref.score) > SCORE_TOL:
                problems.append(f"{it['candidate_id']}, {n_slots} slots: differs from the official pass")
        entry = {"parity_max_abs_diff": worst}
        if spec.batched:
            single = [v.score for v in verdicts_for(scorer, prompts, 1)]
            batched = [v.score for v in verdicts_for(scorer, prompts, len(prompts))]
            entry["batch_max_abs_diff"] = max(abs(a - b) for a, b in zip(single, batched))
            if entry["batch_max_abs_diff"] > 1e-4:
                problems.append(f"{n_slots} slots: batched differs by {entry['batch_max_abs_diff']:.2e}")
        report["n_slots"][n_slots] = entry
        print(f"{n_slots:>2} slots: {entry}")

    tokens = scorer.tokens
    report["yes"] = [tokenizer.decode([i]) for i in tokens.yes_ids]
    report["no"] = [tokenizer.decode([i]) for i in tokens.no_ids]
    report["dropped"] = list(tokens.dropped)
    report["rendered_prompt_example"] = scorer.render(message(items[0])) + ANSWER_PREFIXES[scorer.answer_prefix]
    print(f"Yes set {report['yes']}  No set {report['no']}  dropped {report['dropped']}")
    print(f"rendered prompt:\n{report['rendered_prompt_example']!r}")
    for p in problems:
        print("PROBLEM:", p)
    report["problems"] = problems
    finish("tiny-model smoke test", "FAIL" if problems else "PASS", report, args.report)


if __name__ == "__main__":
    main()
