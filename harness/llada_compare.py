"""Add LLaDA scores to the saved BF16 AR sanity run. Never reruns AR models."""
import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import itertools
import json
import math
from pathlib import Path
import statistics
import time

from .ar import aggregate_events, token_sets
from .data import BANK_MAPPING, SANITY_MAPPING, load_records, read_jsonl
from .evaluate import (check_manifest, confusion, index_rows, join_labels,
                       require_same_ids, roc_curve, score_coverage)
from .prompts import TEMPLATES, encode_prompt, messages
from .run import digest

MODEL = 'GSAI-ML/LLaDA-8B-Instruct'
MASK_ID = 126336
DEFAULT_AR = 'results/bf16/ar-sanity-bf16-20261020-015535'


def load_baseline(bank, ar_dir):
    records, bank_hash = load_records(bank, SANITY_MAPPING, 'sanity')
    manifest = json.loads((Path(ar_dir)/'manifest.json').read_text())
    check_manifest(manifest, records, bank_hash, 'sanity')
    if manifest.get('status') != 'completed' or manifest.get('limit') is not None:
        raise ValueError('use a completed full AR run')
    if manifest.get('compute_dtype') != 'torch.bfloat16' or manifest['effective_config']['load_in_4bit']:
        raise ValueError('use the saved unquantized BF16 baseline')
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


def masked_ids(prompt_ids, slots):
    if slots not in (1, 8, 32):
        raise ValueError('answer slots must be 1, 8 or 32')
    if not prompt_ids or MASK_ID in prompt_ids:
        raise ValueError('expected a nonempty prompt without mask tokens')
    if len(prompt_ids) > 1760:
        raise ValueError('prompt exceeds shared 1760-token limit; no truncation')
    return prompt_ids + [MASK_ID]*slots, len(prompt_ids)


def choose_slots(medians):
    if set(medians) != {1,8,32} or not all(math.isfinite(v) and 0 <= v <= 1.000001 for v in medians.values()):
        raise ValueError('need finite Yes+No medians for slots 1,8,32')
    return min(medians, key=lambda n: (-medians[n], n))


def score_one(model, prompt_ids, slots, sets, device):
    import torch
    ids, pos = masked_ids(prompt_ids, slots)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(x)  # Official bidirectional forward; no causal mask or shift.
    torch.cuda.synchronize(device)
    seconds = time.perf_counter()-start
    logits = output.logits[0,pos,:].float()  # First mask, not final token.
    lp = torch.log_softmax(logits,dim=-1)
    events = aggregate_events(lp[sets['yes']['ids']].tolist(),lp[sets['no']['ids']].tolist())
    return dict(**events,forward_seconds=seconds,input_tokens=len(ids),
                first_mask_position=pos,top_token_id=int(logits.argmax().item()))


def write_rows(path, rows):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row,allow_nan=False)+'\n')


def write_csv(path, rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def compare(records, labels, scores):
    ids=[r['id'] for r in records]
    for name,indexed in scores.items():
        require_same_ids(set(ids),indexed,name)
        for row in indexed.values():
            if row.get('status') != 'completed' or not math.isfinite(row['score']):
                raise ValueError(f'{name}: incomplete or nonfinite scores')
    singles,pairs,review=[],[],[]
    for domain in ('all','math','code'):
        subset=[r['id'] for r in records if domain=='all' or r['domain']==domain]
        ys=[labels[c] for c in subset]
        for name,indexed in scores.items():
            values=[indexed[c]['score'] for c in subset]
            singles.append(dict(model=name,domain=domain,candidates=len(subset),
                                auroc=roc_curve(ys,values)[2],
                                **confusion(ys,[s >= 0 for s in values])))
        for a,b in itertools.combinations(scores,2):
            aa=[scores[a][c]['score'] >= 0 for c in subset]
            bb=[scores[b][c]['score'] >= 0 for c in subset]
            pairs.append(dict(pair=f'{a}+{b}',domain=domain,candidates=len(subset),
                              threshold_a=0,threshold_b=0,
                              **confusion(ys,[x and y for x,y in zip(aa,bb)]),
                              wrong_accepted_a_only=sum(y==0 and x and not z for y,x,z in zip(ys,aa,bb)),
                              wrong_accepted_b_only=sum(y==0 and z and not x for y,x,z in zip(ys,aa,bb)),
                              wrong_rejected_by_both=sum(y==0 and not x and not z for y,x,z in zip(ys,aa,bb))))
    for r in records:
        cid=r['id']
        row=dict(candidate_id=cid,domain=r['domain'],label=labels[cid],
                 problem=r['problem'],candidate=r['candidate_answer'])
        for name in scores:
            row[f'score_{name}']=scores[name][cid]['score']
            row[f'accept_{name}_at_zero']=scores[name][cid]['score'] >= 0
        review.append(row)
    return singles,pairs,review


def analyze(out, records, bank_hash, ar_manifest, ar_scores, bank, ar_dir):
    lm=json.loads((out/'llada_manifest.json').read_text())
    if lm.get('status') != 'completed' or lm['bank_sha256'] != bank_hash:
        raise ValueError('LLaDA run incomplete or input differs')
    if lm['ar_manifest_sha256'] != digest((Path(ar_dir)/'manifest.json').read_bytes()):
        raise ValueError('AR reference manifest changed')
    p=out/'scores_llada.jsonl'
    if digest(p.read_bytes()) != lm['outputs_sha256'][p.name]:
        raise ValueError('LLaDA score checksum mismatch')
    llada=index_rows(read_jsonl(p)[0],'candidate_id',str(p))
    for row in llada.values():
        if (row['bank_sha256'] != bank_hash or row['template'] != 'T1'
                or row['answer_slots'] != lm['answer_slots'] or row['model_revision'] != lm['model_revision']):
            raise ValueError('LLaDA score settings mismatch')
    labels=join_labels(records,read_jsonl(bank)[0],bank_hash)
    scores={'Qwen':ar_scores['A'],'Mistral':ar_scores['B'],'LLaDA':llada}
    singles,pairs,review=compare(records,labels,scores)
    write_csv(out/'single_metrics.csv',singles)
    write_csv(out/'pair_metrics.csv',pairs)
    write_csv(out/'comparison_review.csv',review)
    result=dict(description='Sanity integration comparison; not H1 or research evidence',
                pair_rule='AND at fixed score zero; exploratory, not matched TAR',
                candidates=len(records),bank_sha256=bank_hash,singles=singles,pairs=pairs,
                sanity_auroc_passed={r['model']:r['auroc'] >= .90 for r in singles if r['domain']=='all'},
                llada_checks=lm['checks'],comparable_cost_claim=False,
                cost_note='Saved AR and new LLaDA runs use different environments; no matched-cost claim.')
    (out/'comparison_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# AR + diffusion: comparison on the original 50 sanity candidates','',
           'Reused saved BF16 Qwen/Mistral scores; only LLaDA was run.','',
           '| Verifier | Overall AUROC | Sanity AUROC >= 0.90 |','|---|---:|---|']
    for r in singles:
        if r['domain']=='all': lines.append(f"| {r['model']} | {r['auroc']:.4f} | {r['auroc'] >= .90} |")
    lines+=['','| Pair (AND at score zero) | Correct accepted / 25 | Wrong accepted / 25 | TAR | FAR |',
            '|---|---:|---:|---:|---:|']
    for r in pairs:
        if r['domain']=='all': lines.append(f"| {r['pair']} | {r['tp']} | {r['fp']} | {r['tar']:.3f} | {r['far']:.3f} |")
    lines+=['','These exploratory thresholds do not match pair TAR and cannot establish superiority. '
            'The sanity fixture is not a validation or research test set. Thresholds and prompts '
            'were not fitted to its labels. The existing AR models already fail the overall sanity target.',
            '',f"LLaDA slots: {lm['answer_slots']}, selected on separate dev candidates by median Yes+No mass.",
            f"Reference/repeat checks passed: {lm['checks']['reference_passed']}; full project gates remain pending.",
            '', 'See `single_metrics.csv` for domain AUROCs, `pair_metrics.csv` for all pair counts, '
            'and `comparison_review.csv` for aligned candidates, labels and three scores.',
            '', 'Scientific comparison still requires pilot/validation gates, validation-selected '
            'thresholds for matched TAR, and a held-out evaluation. No cost-matching claim is made.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def run_llada(args, records, bank_hash, ar_manifest):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    if version('transformers') != '4.38.2':
        raise ValueError('use the separate LLaDA environment: transformers==4.38.2')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError('native BF16 CUDA GPU required')
    dev,dev_hash=load_records(args.dev_bank,BANK_MAPPING,'dev_bank')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',model_id=MODEL,template='T1',template_text=TEMPLATES['T1'],
                  bank_sha256=bank_hash,dev_bank_sha256=dev_hash,mask_id=MASK_ID,
                  ar_manifest_sha256=digest((Path(args.ar_run)/'manifest.json').read_bytes()),
                  ar_score_hashes=ar_manifest['outputs_sha256'],started_utc=datetime.now(timezone.utc).isoformat(),
                  gpu=torch.cuda.get_device_name(0),dtype='torch.bfloat16',batch_size=1,
                  packages={p:version(p) for p in ('torch','transformers','accelerate','tokenizers','protobuf')},
                  source_sha256={p.name:digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')})
    def save():
        (out/'llada_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    save()
    try:
        config=AutoConfig.from_pretrained(MODEL,revision=args.revision,trust_remote_code=True)
        revision=getattr(config,'_commit_hash',None)
        if not revision: raise ValueError('exact LLaDA commit could not be resolved')
        manifest['model_revision']=revision;save()
        tok=AutoTokenizer.from_pretrained(MODEL,revision=revision,trust_remote_code=True)
        model=AutoModel.from_pretrained(MODEL,revision=revision,config=config,trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to('cuda:0').eval()
        device=torch.device('cuda:0');sets=token_sets(tok)
        if tok.mask_token_id is not None and tok.mask_token_id != MASK_ID:
            raise ValueError('unexpected mask token ID')
        if not isinstance(tok.chat_template, str) or not tok.chat_template.strip():
            raise ValueError('LLaDA tokenizer must supply its official chat template')
        if tok.convert_tokens_to_ids('<|mdm_mask|>') != MASK_ID:
            raise ValueError('official mask token does not map to 126336')
        manifest.update(token_sets=sets,chat_template=tok.chat_template)
        prompts={r['id']:encode_prompt(tok,r,'T1') for r in records+dev}
        for ids in prompts.values(): masked_ids(ids,32)
        rendered=tok.apply_chat_template(messages(records[0],'T1'),tokenize=False,add_generation_prompt=True)
        if tok.encode(rendered,add_special_tokens=False) != prompts[records[0]['id']]:
            raise ValueError('chat template encoding mismatch')
        manifest['rendered_prompt_example']=rendered;save()
        medians={}
        for slots in (1,8,32):
            rows=[]
            for i,r in enumerate(dev):
                result=score_one(model,prompts[r['id']],slots,sets,device)
                rows.append(dict(candidate_id=r['id'],bank_sha256=dev_hash,model_id=MODEL,
                                 model_revision=revision,template='T1',answer_slots=slots,
                                 status='completed',cold=i==0,**result))
            write_rows(out/f'dev_slots{slots}.jsonl',rows)
            medians[slots]=statistics.median(r['yes_no_mass'] for r in rows)
            print(f'Development slots={slots}: median Yes+No mass={medians[slots]:.6f}',flush=True)
        slots=choose_slots(medians)
        manifest.update(answer_slots=slots,dev_median_yes_no_mass=medians,
                        slot_selection='highest median Yes+No mass on dev; ties favor fewer slots');save()
        rows=[]
        for i,r in enumerate(records):
            result=score_one(model,prompts[r['id']],slots,sets,device)
            rows.append(dict(candidate_id=r['id'],id=r['id'],bank_sha256=bank_hash,model='LLaDA',
                             model_id=MODEL,model_revision=revision,domain=r['domain'],template='T1',
                             answer_slots=slots,status='completed',cold=i==0,**result))
            print(f"LLaDA {i+1}/{len(records)} {r['id']}: score={result['score']:.4f}",flush=True)
        write_rows(out/'scores_llada.jsonl',rows)
        checks=[];byid={r['candidate_id']:r for r in rows}
        reference_items=[r for d in ('math','code') for r in [x for x in records if x['domain']==d][:10]]
        for r in reference_items:
            ids,pos=masked_ids(prompts[r['id']],slots)
            with torch.inference_mode():
                logits=model(torch.tensor([ids],device=device)).logits[0,pos,:].float()
                lp=torch.log_softmax(logits,dim=-1)
                score=(torch.logsumexp(lp[sets['yes']['ids']],0)-torch.logsumexp(lp[sets['no']['ids']],0)).item()
            delta=abs(score-byid[r['id']]['score'])
            top_matches=int(logits.argmax().item())==byid[r['id']]['top_token_id']
            checks.append(dict(candidate_id=r['id'],score_delta=delta,top_token_matches=top_matches,
                               passed=delta <= 1e-3 and top_matches))
        write_rows(out/'reference_checks.jsonl',checks)
        manifest.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat(),
                        checks=dict(reference_passed=all(r['passed'] for r in checks),reference_prompts=len(checks),
                                    reference_tolerance=1e-3,method='repeat official remote-model forward; independent Torch reduction',
                                    batching_check='pending; scoring is unbatched',full_project_gates_passed=False),
                        outputs_sha256={p.name:digest(p.read_bytes()) for p in out.glob('*.jsonl')});save()
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');save();raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',default='bank/sanity/sanity_set.jsonl')
    p.add_argument('--ar-run',default=DEFAULT_AR)
    p.add_argument('--dev-bank',default='bank/dev/dev_bank.jsonl')
    p.add_argument('--revision',default='main')
    p.add_argument('--output-dir')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--analyze-only',action='store_true')
    args=p.parse_args(argv)
    records,bank_hash,manifest,scores=load_baseline(args.bank,args.ar_run)
    if args.validate_only:
        dev,_=load_records(args.dev_bank,BANK_MAPPING,'dev_bank')
        print(f'Validated existing BF16 AR scores: {len(records)} candidates. Separate dev inputs: {len(dev)}. No AR inference needed.')
        return 0
    if not args.output_dir: p.error('--output-dir required')
    if not args.analyze_only: run_llada(args,records,bank_hash,manifest)
    analyze(Path(args.output_dir),records,bank_hash,manifest,scores,args.bank,args.ar_run)
    return 0

if __name__=='__main__': raise SystemExit(main())
