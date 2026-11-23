# Next steps

**Start here.** The plan is [PLAN.md](PLAN.md) (version 1.5). The pilot and the midterm report are done. What's left is the full experiment, the presentation and the final report. It is split into **six packages, one per person**. There are no internal dates: do each step as soon as its inputs exist.

## The rules are settled (PLAN §2, §4, §7; records in `decisions/`)

| Question | Rule |
|---|---|
| Which wordings must a verifier be stable on? | **T1 and T2** (AUROC spread ≤ 0.05). T3 is only reported. |
| Is the 0.90 sanity bar still a gate? | **No**, it is a smoke test. Gate A on the pilot decides competence. |
| Where do new math problems come from? | Unused GSM8K-Platinum and MATH-500, plus **the rest of the MATH test set**. Half GSM8K, half MATH, as in the pilot. |
| What does "best" mean in the dynamic-range gate? | The verifier that **accepts the fewest wrong answers** at 90% TAR. |
| What is the main test? | **One test (PLAN §5): math.** The best of LLaDA+Qwen, LLaDA+Llama, LLaDA+Gemma against the best of Qwen+Llama, Qwen+Gemma, Llama+Gemma. Code is reported separately. |
| When is go/no-go decided? | **Once, from the pilot, before any full-bank answer is scored** (PLAN §7). It needs the pilot label audit, test 4 in math and the AR repeat-run check. |
| Who checks the labels? | **Codex, blind, plus one person** (PLAN §2). Math only decides ([record](decisions/2026-11-22-label-check-by-domain.md), post hoc): no error in the pilot's 100 math items; at most 3 in 200 math items of the full bank. Code is HumanEval+ only, with no pass mark. |
| When does the final comparison count as matched? | Each pair's test TAR **within 5 points of 90%**, the two pairs **within 3 points of each other** (PLAN §5 step 7). |
| When are licenses and terms checked? | **Before the release.** Fields, stated licenses and hashes are recorded at download. |
| What if LLaDA fails test 4 in math? | Its reconstruction score if that passes on the pilot; otherwise the AR-only study. |
| What if LLaDA is below the bar on validation? | Reported. It does not change the test. |
| What if an AR verifier fails test 4 in math? | It drops out. Fewer than three left: C also scores Qwen3-8B, for the AR+AR pairs only. No cross-family AR+AR pair left: H1 is not run. |

The three wordings (`harness/prompts.py`): **T1** "Is the final answer correct?", **T2** the same question to "a grader", **T3** "Is the final answer incorrect?".

## Order of work

```
0  GO/NO-GO (first):  notebook 09: test 4 for every verifier + AR repeat runs }  -> F writes the FINAL MODEL RECORD in decisions/, team merges it
                      label check: Codex + one person, 200 pilot labels     }
   Meanwhile, in parallel:  A math bank   B code bank   D reconstruction score   E analysis code   F slides, report skeleton

1  A + B finish the banks  ->  Codex + one person check 200  ->  F does the BANK FREEZE (split fixed, test labels to F)
2  C + D score both banks (validation and test) with T1 and T2    [needs the final model record and the bank freeze]
3  E picks pairs and thresholds on VALIDATION only, reports LLaDA's re-check  ->  E makes the PREREGISTRATION COMMIT, C reviews it
4  F joins the test labels, once  ->  E runs the final test, once
5  Everyone writes; F assembles the slides and the report
```

After step 0, nothing changes the models, the domain or the score. If step 0 says "repair", D's reconstruction score becomes part of step 2.

## The six packages

### A. Math bank
1. Get the new MATH problems (PLAN §2): `math_splits/train.jsonl` from OpenAI's `prm800k`, rows whose `unique_id` starts with `test/`. Record the fields, the stated license and the hash, and drop anything already in MATH-500 or the pilot, by id and by text.
2. Select problems: all 1,059 unused GSM8K-Platinum, all 239 unused MATH-500, a seeded draw (seed 544) of 500 from the new set with numeric answers. Start from `bank/select_pilot.py`.
3. Generate (4 generators, 4 samples each), label and pair, reusing `bank/generate.py`, `bank/generate_api.py`, `bank/label.py`, `bank/make_pairs.py`. The API budget cap in `generate_api.py` defaults to $5 and the pilot already spent $1.19, so raise it or use a new ledger. Run `bank/token_lengths.py` with every verifier's tokenizer loaded, plus Qwen3-8B's (Llama and Gemma failed to load for the pilot).
4. Draw the bank by PLAN §2's draw rule. **Careful:** `bank/sample_pilot.py` takes a fixed target and splits evenly over *sources*, so with three sources GSM8K would get only a third; group MATH-500 and the new problems, take MATH-500 first, and compute the target from the rule.
5. Split by problem, stratified by source: 1/3 validation, 2/3 test.

**Done when:** a hashed math bank of about 2,100 answers exists with separate validation and test label files, and the test labels are handed to F and deleted from your copies.

### B. Code bank and checks
1. The **answer-swap items** for test 4 (PLAN §4) are built: `bank/pilot/swap/` (`bank/answer_swap.py`; 100 math and 100 code items, originals and swaps in one file, the key beside it).
2. Build the code bank from the 114 unused HumanEval+ problems (8 samples, 4 generators), reusing `bank/label_evalplus.py`. MBPP+ is left out. Split like math.
3. Label check (PLAN §2): the pilot's is done (math passed, `bank/pilot/audit/`); 200 new math labels from the full bank are checked before the freeze. `bank/codex_audit.py` gets Codex's blind verdicts, `bank/audit_sheet.py` writes the sheet one person reads and applies the rule. On the full bank B reads and A runs the commands ([record](decisions/2026-11-22-label-keeper-handover.md)). Pass: none in the pilot's 100 math items; at most 3 in the full bank's 200.
4. With A and F: tag 200 wrong math answers from the test split by error type (PLAN §2), before deleting your copies. Code error types come from the execution logs. Tags and logs go to F with the test labels.

**Done when:** a hashed code bank of about 300 answers exists, and both audits and the error tags are recorded.

### C. AR scoring
1. Run **notebook 09** on an A100: test 4 (answer swap) for all eight verifiers and the repeat-run check (scoring test 3) for the AR models, the H1 models first. `analysis/step0.py` applies the rules per domain.
2. Commit `results/step0/` and give F the go/no-go table.
3. Score **both splits** of both banks with T1 and T2: Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B, Mistral-7B (H2 only), Llama-3.2-3B (gate A's reference and H2) and Qwen2.5-1.5B (H2 only). Use **notebook 13** (`harness/full_run.py`, `configs/full_*.json`), from one Google account for C and D together. All on **one GPU type**, A100 or L4, whichever the first scoring run gets ([record](decisions/2026-11-22-scoring-one-gpu-type.md)), batch size 1; Gemma with eager attention, the rest SDPA.
4. AR arm of the reconstruction score: exact log-probability of the answer, for the four AR models, on the same answers as D's step 4.
5. Only if the final model record leaves fewer than three AR verifiers: score Qwen3-8B (thinking off) too, with the rest.
6. Reproduce the pilot gates and, at the end, the final main table from the saved outputs, and review E's preregistration commit.

**Done when:** complete score files for 6 models × 2 templates × 2 banks, test 4 results for each model, and the reproduction check.

### D. Diffusion scoring and reconstruction
1. Test 4 for LLaDA and Dream runs in notebook 09 (C); check their environments and score files.
2. Implement the **reconstruction score** (PLAN §4): LLaDA's `get_log_likelihood` Monte Carlo estimate, ported to Dream with its shift. The side experiment uses 128 samples ([record](decisions/2026-11-22-reconstruction-128-samples.md), post hoc). Run it on the pilot and report gate A (per token, 128 samples, seed 544) and the repeat check with seed 545 (PLAN §7).
3. Score both splits of both banks with T1 and T2 for LLaDA and Dream: unbatched, in notebook 13, on the **same GPU type** as the AR models ([record](decisions/2026-11-22-scoring-one-gpu-type.md); the pilot's Dream run used an L4, so its time is not comparable).
4. Run the reconstruction score on 500 validation and 500 test answers of the math bank (on the whole math bank only if it is needed as the repair).

**Done when:** complete score files for both models, and reconstruction results on the pilot.

### E. Analysis
Write and test everything on the pilot and synthetic data **now**, so it runs the moment scores arrive. Start from `analysis/metrics.py` and `analysis/pilot.py`.
1. Recompute the pilot precision check (PLAN §6) with committed code.
2. Validation step: report LLaDA's re-check (it does not change the test); choose the best AR+AR and AR+diffusion pair from the six in PLAN §5; set every threshold; cost check; pseudo-pairs; best single verifier; thresholds at 80%, 90% and 95% pair TAR; selection stability over validation resamples; the analysis manifest. Commit it yourself; C reviews, A and B do not. (`analysis/pilot.py` still uses T1–T3 for its decision and highest AUROC for the dynamic range; use the settled rules.)
3. Test step: ΔFAR with a paired bootstrap over whole problems (10,000 replicates, stratified by domain), matching check, decision table.
4. H2: κ, Spearman, CAPA, and the trend across AR pairs. E1: Dream+Qwen. Single-verifier ROC/AUROC, calibration (ECE, reliability) and GPU-seconds; accuracy-versus-cost curves. Breakdowns by source, generator and error type. Figures.

**Done when:** the frozen manifest is committed, then the final tables and figures are produced from it in one run.

### F. Custody, infrastructure and writing
1. Do the bank freeze: each bank file is committed once (notebook 13 refuses a bank that changed after its first commit). Keep the **test labels**, code execution logs and error tags off the repo and the shared storage used by C, D and E. Join the labels once, after E's preregistration commit, and log it.
2. The **final model record** (PLAN §7) is written: go. Finish `decisions/`: confirm the post-pilot records, the pilot status, and the instructor note (gate B, MATH problems, code descriptive and HumanEval+ only, the per-domain label check, Milestone 2 late).
3. Check every citation, including PRISM, DEER, HIVE and FreeCache (PLAN §14), and make the difference from prior work explicit.
4. Presentation (5 slides, 6 minutes), then the report (8 pages + appendix). Build both skeletons now with the pilot numbers.
5. Check the terms of every data source and generator (PLAN §9), then release the labeled bank and analysis code.

**Done when:** the presentation is delivered, the report is submitted, and the release is out.

## Rules everyone follows
- **Only validation labels may be used to choose anything.** Test labels are held by F and joined once, after the freeze commit.
- **A and B take no part in selection or threshold work**, and delete their copies of the test labels after handing them to F.
- **Never commit the full bank's `labeled.jsonl` or test labels** before the join. (The pilot's are in the repo; that one is exploratory.)
- **No T3 on the full bank.** No new prompt wordings, gates or model swaps (Qwen3-8B only under the rule above). Until the final model record is merged, a change to the plan needs a dated file in `decisions/`. After it, the final test does not change.
- The pilot's scores are in-sample. Never call them H1 results.
- Optional, only if someone is free after the main result: critique-before-verdict, the Qwen3-8B check (unless the rule above requires it), answer-only control.

## Status

| Work | Status |
|---|---|
| Pilot, midterm report | Done |
| Rules settled | Written; confirmed when this is merged |
| Go/no-go: pilot label audit, test 4, final model record | **Go** ([final model record](decisions/2026-11-22-final-model-record.md), `configs/final_models.json`) |
| Reconstruction score | Run on the pilot (notebook 12): passes gate A, misses the repeat check (0.949 vs 0.95); repair not needed. Side experiment run, exploratory, validation half only: AUROC 0.654, not clearly above LLaDA's verdict score ([results](results/reconstruction/README.md)) |
| Math bank, code bank | Built: 2,100 math answers and 350 HumanEval+ code answers, split. Full-bank math label check passed (0 errors, 105 of 200 read). Error tags under way; then the bank freeze |
| Scoring the full bank | Done: 8 verifiers, T1 and T2, both banks, A100 (`results/full/`) |
| Validation, preregistration, final test | Validation done (`results/final/validation.md`); preregistration under review; final test after it is merged |
| Presentation | Slot not confirmed |
| Final report | Not started |
