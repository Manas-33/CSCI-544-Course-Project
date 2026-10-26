"""Compare saved LLaDA sanity scores with the saved AR sanity run. No model runs here.

Scoring goes through harness/diffusion (run_score.py, checks/). The committed LLaDA run in
results/llada-comparison-20261019-213107 was scored by an earlier version of this file
(commit 030a84a); its llada_manifest.json records how. This module now only:

- compares those LLaDA scores with the BF16 + SDPA Qwen/Mistral run (single and AND-pair
  metrics at the exploratory zero threshold), and
- checks a new LLaDA score file from harness/diffusion against them (--parity), so both
  scorers are known to agree before the old one's results are relied on.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

from .data import SANITY_MAPPING, load_records, read_jsonl
from .evaluate import (check_manifest, confusion, index_rows, join_labels,
                       require_same_ids, roc_curve, score_coverage)
from .run import digest

DEFAULT_AR = 'results/bf16/sdpa/ar-sanity-bf16-sdpa-20261026-000459'
DEFAULT_LLADA = 'results/llada-comparison-20261019-213107'
# Agreement between two LLaDA score files, using PLAN.md §4 test 3's bounds: |Δs| < 0.05
# (its batched-vs-unbatched bound) and Spearman > 0.999 (its repeat-run bound).
MAX_SCORE_DELTA = 0.05
MIN_SPEARMAN = 0.999


def load_baseline(bank, ar_dir):
    records, bank_hash = load_records(bank, SANITY_MAPPING, 'sanity')
    manifest = json.loads((Path(ar_dir)/'manifest.json').read_text())
    check_manifest(manifest, records, bank_hash, 'sanity')
    if manifest.get('status') != 'completed' or manifest.get('limit') is not None:
        raise ValueError('use a completed full AR run')
    if manifest.get('compute_dtype') != 'torch.bfloat16' or manifest['effective_config']['load_in_4bit']:
        raise ValueError('use the saved unquantized BF16 baseline')
    if manifest.get('attention_implementation') != 'sdpa':
        raise ValueError('use an SDPA AR run; the eager-attention BF16 run gave wrong Qwen scores')
    if len(records) != 50:
        raise ValueError('expected all 50 original sanity candidates')
    scores = {}
    for name in ('A', 'B'):
        p = Path(ar_dir)/f'scores_{name}.jsonl'
        if digest(p.read_bytes()) != manifest['outputs_sha256'][p.name]:
            raise ValueError(f'AR score checksum mismatch: {p}')
        scores[name] = index_rows(read_jsonl(p)[0], 'candidate_id', str(p))
        if score_coverage(records, scores[name], bank_hash, name)['completed'] != len(records):
            raise ValueError(f'{name}: incomplete baseline')
    return records, bank_hash, manifest, scores


def load_llada(run_dir, bank_hash):
    run_dir = Path(run_dir)
    lm = json.loads((run_dir/'llada_manifest.json').read_text())
    if lm.get('status') != 'completed' or lm['bank_sha256'] != bank_hash:
        raise ValueError('LLaDA run incomplete or input differs')
    p = run_dir/'scores_llada.jsonl'
    if digest(p.read_bytes()) != lm['outputs_sha256'][p.name]:
        raise ValueError('LLaDA score checksum mismatch')
    llada = index_rows(read_jsonl(p)[0], 'candidate_id', str(p))
    for row in llada.values():
        if (row['bank_sha256'] != bank_hash or row['template'] != 'T1'
                or row['answer_slots'] != lm['answer_slots'] or row['model_revision'] != lm['model_revision']):
            raise ValueError('LLaDA score settings mismatch')
    return lm, llada


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def compare(records, labels, scores):
    ids = [r['id'] for r in records]
    for name, indexed in scores.items():
        require_same_ids(set(ids), indexed, name)
        for row in indexed.values():
            if row.get('status') != 'completed' or not math.isfinite(row['score']):
                raise ValueError(f'{name}: incomplete or nonfinite scores')
    singles, pairs, review = [], [], []
    for domain in ('all', 'math', 'code'):
        subset = [r['id'] for r in records if domain == 'all' or r['domain'] == domain]
        ys = [labels[c] for c in subset]
        for name, indexed in scores.items():
            values = [indexed[c]['score'] for c in subset]
            singles.append(dict(model=name, domain=domain, candidates=len(subset),
                                auroc=roc_curve(ys, values)[2],
                                **confusion(ys, [s >= 0 for s in values])))
        for a, b in itertools.combinations(scores, 2):
            aa = [scores[a][c]['score'] >= 0 for c in subset]
            bb = [scores[b][c]['score'] >= 0 for c in subset]
            pairs.append(dict(pair=f'{a}+{b}', domain=domain, candidates=len(subset),
                              threshold_a=0, threshold_b=0,
                              **confusion(ys, [x and y for x, y in zip(aa, bb)]),
                              wrong_accepted_a_only=sum(y == 0 and x and not z for y, x, z in zip(ys, aa, bb)),
                              wrong_accepted_b_only=sum(y == 0 and z and not x for y, x, z in zip(ys, aa, bb)),
                              wrong_rejected_by_both=sum(y == 0 and not x and not z for y, x, z in zip(ys, aa, bb))))
    for r in records:
        cid = r['id']
        row = dict(candidate_id=cid, domain=r['domain'], label=labels[cid],
                   problem=r['problem'], candidate=r['candidate_answer'])
        for name in scores:
            row[f'score_{name}'] = scores[name][cid]['score']
            row[f'accept_{name}_at_zero'] = scores[name][cid]['score'] >= 0
        review.append(row)
    return singles, pairs, review


def analyze(run_dir, records, bank_hash, ar_manifest, ar_scores, bank, ar_dir):
    out = Path(run_dir)
    lm, llada = load_llada(out, bank_hash)
    labels = join_labels(records, read_jsonl(bank)[0], bank_hash)
    scores = {'Qwen': ar_scores['A'], 'Mistral': ar_scores['B'], 'LLaDA': llada}
    singles, pairs, review = compare(records, labels, scores)
    write_csv(out/'single_metrics.csv', singles)
    write_csv(out/'pair_metrics.csv', pairs)
    write_csv(out/'comparison_review.csv', review)
    ar_manifest_sha = digest((Path(ar_dir)/'manifest.json').read_bytes())
    result = dict(description='Sanity integration comparison; not H1 or research evidence',
                  pair_rule='AND at fixed score zero; exploratory, not matched TAR',
                  candidates=len(records), bank_sha256=bank_hash,
                  ar_run=ar_dir, ar_manifest_sha256=ar_manifest_sha,
                  ar_attention=ar_manifest['attention_implementation'], ar_compute_dtype=ar_manifest['compute_dtype'],
                  llada_run=str(run_dir), llada_model_revision=lm['model_revision'], llada_answer_slots=lm['answer_slots'],
                  singles=singles, pairs=pairs,
                  sanity_auroc_passed={r['model']: r['auroc'] >= .90 for r in singles if r['domain'] == 'all'},
                  llada_repeat_check=dict(passed=lm['checks']['reference_passed'], prompts=lm['checks']['reference_prompts'],
                                          note='The same forward pass run twice; not PLAN.md §4 test 2 (reference parity).'),
                  llada_batching_check=lm['checks']['batching_check'],
                  comparable_cost_claim=False,
                  cost_note='AR and LLaDA ran in different environments and on different A100s; no matched-cost claim.')
    (out/'comparison_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# AR + diffusion: comparison on the 50 sanity candidates', '',
             f'Qwen/Mistral: the saved BF16 + SDPA run `{ar_dir}`. LLaDA: its own saved run `{run_dir}` '
             f'({lm["answer_slots"]} answer slot, T1). Nothing was rescored for this comparison.', '',
             '| Verifier | Overall AUROC | Math AUROC | Code AUROC | Sanity AUROC ≥ 0.90 |', '|---|---:|---:|---:|---|']
    by = {(r['model'], r['domain']): r for r in singles}
    for name in scores:
        lines.append(f"| {name} | {by[name, 'all']['auroc']:.3f} | {by[name, 'math']['auroc']:.3f} | "
                     f"{by[name, 'code']['auroc']:.3f} | {'PASS' if by[name, 'all']['auroc'] >= .90 else 'FAIL'} |")
    lines += ['', '| Pair (AND at score zero) | Correct accepted / 25 | Wrong accepted / 25 | TAR | FAR |',
              '|---|---:|---:|---:|---:|']
    for r in pairs:
        if r['domain'] == 'all':
            lines.append(f"| {r['pair']} | {r['tp']} | {r['fp']} | {r['tar']:.3f} | {r['far']:.3f} |")
    lines += ['', 'The zero threshold gives each pair a different TAR, so these FARs cannot be compared with each other '
              'and establish nothing about H1. The sanity fixture is a scoring check, not a validation or test set; '
              'no threshold or prompt was fitted to its labels.', '',
              f"LLaDA's answer-slot count was chosen on the development bank by median Yes+No mass. "
              f"Its run's repeat check (the same forward pass run twice on {lm['checks']['reference_prompts']} items) "
              f"passed: {lm['checks']['reference_passed']}. Still pending: PLAN.md §4 test 2 (reference parity against "
              "the official model code, `harness/diffusion/checks/check_parity.py`), test 3's batch invariance, and "
              "agreement between this run and the shared scorer `harness/diffusion` (`--parity`).", '',
              'See `single_metrics.csv`, `pair_metrics.csv` and `comparison_review.csv` for every count and score.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


def spearman(xs, ys):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0]*len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                result[order[k]] = (i + j)/2 + 1
            i = j + 1
        return result
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx)/len(rx), sum(ry)/len(ry)
    cov = sum((a - mx)*(b - my) for a, b in zip(rx, ry))
    spread = math.sqrt(sum((a - mx)**2 for a in rx)*sum((b - my)**2 for b in ry))
    if not spread:
        raise ValueError('Spearman is undefined for constant scores')
    return cov/spread


def parity(new_rows, lm, llada, bank_hash):
    """Agreement of harness/diffusion LLaDA score rows with the committed LLaDA scores."""
    new = index_rows(new_rows, 'candidate_id', 'parity scores')
    require_same_ids(set(llada), new, 'parity scores')
    for cid, row in new.items():
        if row.get('bank_hash') != bank_hash:
            raise ValueError(f"{cid}: bank_hash {row.get('bank_hash')} is not the sanity set's {bank_hash}")
        if row.get('status') != 'ok' or row.get('model') != 'llada' or row.get('template') != 'T1':
            raise ValueError(f'{cid}: expected an ok T1 LLaDA row from harness/diffusion')
        if row.get('n_slots') != lm['answer_slots']:
            raise ValueError(f"{cid}: n_slots {row.get('n_slots')} differs from the saved run's {lm['answer_slots']}")
        if row.get('revision') != lm['model_revision']:
            raise ValueError(f"{cid}: revision {row.get('revision')} differs from the saved run's {lm['model_revision']}")
    ids = sorted(llada)
    old, cur = [llada[c]['score'] for c in ids], [new[c]['score'] for c in ids]
    deltas = [abs(a - b) for a, b in zip(old, cur)]
    rho = spearman(old, cur)
    worst = max(range(len(ids)), key=lambda i: deltas[i])
    return dict(items=len(ids), max_abs_score_delta=max(deltas), worst_item=ids[worst],
                spearman=rho, same_verdict_at_zero=sum((a >= 0) == (b >= 0) for a, b in zip(old, cur)),
                max_allowed_delta=MAX_SCORE_DELTA, min_spearman=MIN_SPEARMAN,
                passed=max(deltas) < MAX_SCORE_DELTA and rho > MIN_SPEARMAN)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--bank', default='bank/sanity/sanity_set.jsonl')
    p.add_argument('--ar-run', default=DEFAULT_AR)
    p.add_argument('--llada-run', default=DEFAULT_LLADA, help='saved LLaDA run; the comparison is written into it')
    p.add_argument('--validate-only', action='store_true', help='check the saved inputs, write nothing')
    p.add_argument('--parity', metavar='SCORES_JSONL',
                   help='harness/diffusion LLaDA score rows to check against the saved LLaDA run')
    args = p.parse_args(argv)
    records, bank_hash, manifest, scores = load_baseline(args.bank, args.ar_run)
    lm, llada = load_llada(args.llada_run, bank_hash)
    if args.validate_only:
        print(f'Validated the SDPA AR run and the saved LLaDA run: {len(records)} candidates each.')
        return 0
    if args.parity:
        result = parity(read_jsonl(args.parity)[0], lm, llada, bank_hash)
        print(json.dumps(result, indent=2))
        return 0 if result['passed'] else 1
    analyze(args.llada_run, records, bank_hash, manifest, scores, args.bank, args.ar_run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
