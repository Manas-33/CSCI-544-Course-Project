# Analysis

Reads saved scores and labels; never runs a model. Standard library, numpy and matplotlib only.

| File | What it does |
|---|---|
| `metrics.py`, `pilot.py` | Basic metrics; the pilot's gates and tables (its decision still uses T1–T3 and the highest AUROC: the settled rules are in PLAN §5 and §7) |
| `final_settings.py` | Every fixed number (`RULES`), the verifiers, and every pair that is analyzed |
| `final_data.py` | Banks, scores and validation labels, with all checks |
| `final_stats.py` | Problem bootstrap, thresholds, matching check, decision table |
| `final_validation.py` | Validation step: thresholds, selection, cost check, LLaDA's re-check, the analysis manifest |
| `final_evaluate.py` | Test step, run once: final test, reference rows, secondary results, reading of the outcome |
| `overlap.py` | H2 and E1: κ, Spearman, CAPA, competence trend |
| `final_report.py` | Tables and figures from `results.json` (computes nothing) |
| `final_pilot.py`, `final_synthetic.py` | Smoke test on the pilot (in-sample, never an H1 result); synthetic study for the tests |
| `precision.py` | PLAN §6 from the pilot, and the matching-check simulation |

## Commands, in order

Run from the repository root.

**0. Any time.** Tests, the pilot smoke test, and the precision check (before the bank freeze):

```bash
python -m pytest -q analysis
python -m analysis.final_pilot --out results/final/pilot_smoke
python -m analysis.precision --out results/final/precision
```

**1. After the final model record is merged** (PLAN §7): copy `configs/final_models.example.json` to
`configs/final_models.json` and set what the record says.

| Record says | `mode` | `h1_ar` | `llada_score` | Also |
|---|---|---|---|---|
| Go | `go` | all three | `verdict` | |
| LLaDA fails, reconstruction passes | `repair` | all three | `reconstruction` | `reason` |
| LLaDA fails, reconstruction fails or is late | `pivot` | those left | `null` | `reason`; LLaDA in `dropped` with its reason |
| An AR verifier fails | `go` or `repair` | those left | as above | the dropped ones in `dropped` with a reason; `qwen3_scored: true` (or Qwen3-8B in `dropped`, if it failed its own checks) |
| No cross-family AR+AR pair left | `pivot` | those left | `null` | `reason` |
| Second audit fails | `stopped` | | `null` | `reason`; nothing is analyzed |

**2. Validation step**, when scores for every verifier, T1 and T2, both splits of both banks are in (notebook 13;
`python -m harness.full_run check` says what is missing). Reads validation
labels only, and writes `results/final/analysis_manifest.json` and `validation.md`:

```bash
python -m analysis.final_validation --settings configs/final_models.json --banks bank/full \
    --scores results/full/math results/full/code --out results/final
python -m analysis.final_validation --check results/final/analysis_manifest.json
```

In repair mode, add the file with LLaDA's reconstruction scores of the whole math bank (seed 544, 128 samples) to
`--scores` if it is kept elsewhere. The reconstruction side experiment (PLAN §4, math bank only) is frozen with the same manifest: add
`--side <thresholds from analysis.reconstruction validation> --side-scores <its score files>`. The test step then applies
it after the join.

**3. Preregistration.** Commit the manifest, `configs/final_models.json` and `analysis/` together; C reviews. From here
the analysis code in the manifest is frozen (`final_report.py` is not in it, so a label on a figure can still be fixed).

**4. Test step, once.** F supplies the test labels. The step refuses to run unless the manifest and the code are committed
and everything recomputes to the same manifest, then logs the join in `results/final/join_log.jsonl`:

```bash
python -m analysis.final_evaluate --manifest results/final/analysis_manifest.json \
    --test-labels math=bank/full/math/labels_test.jsonl code=bank/full/code/labels_test.jsonl \
    --out results/final/test
```

Error types are read from the label rows (`error_type`). The math tags must be in the label files, and their hashes in
the bank manifests, before step 2, because the analysis manifest records the bank manifest's hash.

**Commit and push `results/final/join_log.jsonl` right after the run**: until then only this working copy knows about the
join. After that the step looks at every committed version on every branch it can see, and refuses to go on when a recorded
join is missing. There is one log for the whole repository, whatever `--out` a run used. The same command with another
`--out` reproduces the numbers (logged as a rerun). A different manifest, different labels or changed code are refused,
also after a crash: a fix to the frozen code after the join needs a decision record.

## Rules the code keeps

- Thresholds, pairs and weights come from validation and the bank. The only test-tuned numbers are in the row
  "ideal thresholds (test-tuned)", which no verdict uses.
- The bootstrap resamples whole problems, 10,000 replicates, seed 544. Validation resamples use a separate stream
  (`[544, 1]`), so the two are independent.
- Missing, failed, duplicate or misaligned scores are errors. So are a second model revision and scoring settings other
  than the pilot's (BF16, batch size 1, SDPA or Gemma's eager attention, one answer slot). Every diffusion row must come from a finished,
  recorded run; a file resumed after a crash is scored again in one go.
- "On the AR trend", "below gate A's bar" and "beats the pseudo-pair" mean what `decisions/2026-11-21-mechanism-wording.md`
  says.
- CAPA was checked against the paper on 2026-11-21: arXiv:2502.04313v2, Section 2.2, Eqs. 2–4 were read and match the formula
  in `overlap.py` (κ_p, c_obs, c_exp with 1/(options − 1) averaged over items, and the two-option reduction). Appendix A.1 was
  not read word for word.
- A cost is left out when its scores ran on more than one GPU type.
