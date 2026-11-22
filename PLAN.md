# Plan — Do Diffusion Language Models Provide Useful Verification Diversity?

**CSCI 544 (Applied NLP, USC, Fall 2026) · Team "project 16" · Research-focused · Version 1.4, November 21, 2026 (updated after the pilot)**

This is the team's working plan. It is built from [PROJECT.md](PROJECT.md) (the project idea and design) and the execution rules of the earlier team plan, follows the instructor's feedback on the proposal (pairs and thresholds chosen on validation data and frozen before the test set; kept with the team's course records), and adds only improvements that independent reviews confirmed fix real problems. Every difference from PROJECT.md is listed in §13 with its reason. Course deadlines are fixed (§10); everything else is done as soon as possible, in the order in [NEXT_STEPS.md](NEXT_STEPS.md), which splits the remaining work into six packages.

**What changed in version 1.4.** Changes made after the pilot and before any full-bank answer exists: the matching check allows 5 points from 90% and 3 between the pairs (§5 step 7, post hoc), the label check is done by Codex plus one person (§2), the license and terms check comes before the release instead of before generating (§9), the full bank's details are fixed (§2), and how the mechanism and side results are worded is settled without touching the H1 verdict (§4, §5, §7, §8). See `decisions/` and §13 rows 38–42.

**What changed in version 1.3.** The pilot has run. Six rules change after seeing its results, so all six are post hoc and reported as such: gate B uses T1–T2 (§7), the sanity set is a smoke test instead of a gate (§4), the math bank adds problems from the rest of the MATH test set (§2), H1 is fixed to math (§5), LLaDA's validation re-check is reported rather than applied (§7), and cost matching no longer replaces H1 (§5). Terms that were undefined are settled in §7 and §4: the dynamic-range gate's "best" verifier, the one-sided 0.10 rule in gate A, how test 4 is built and judged, and how the reconstruction repair is applied. **There is now one final test (§5), and the go/no-go that decides it is made once, on the pilot, before any full-bank answer is scored (§7).** Cut-order items (1)–(4) are applied (§9). Facts the pilot changed (generators, iLLaDA, unbatched diffusion scoring) are brought up to date. Every change is listed in §13 rows 29–37. The question, the hypotheses, choosing pairs and thresholds on validation, the single test run and the decision rule are unchanged.

Notation: **(est.)** marks a planning estimate to be replaced by a measurement. **Phase 0** is the setup work that came before the pilot.

---

## 1. Question

**At a matched acceptance rate for correct answers and comparable inference cost, does an AR+diffusion verifier pair falsely accept fewer wrong answers than the strongest cross-family AR+AR pair?**

- **Why it matters.** LLMs are used to verify other models' outputs, and ensembles of verifiers help less than expected because strong autoregressive (AR) models make correlated mistakes. Diffusion language models learn by global denoising rather than left-to-right prediction, an objective-level difference that could produce different mistakes. To our knowledge, no published work tests diffusion models as verifiers of AR outputs against matched AR baselines (re-check before each report).
- **A negative result is a valid outcome.** "Changing model class buys no complementarity beyond strong cross-family AR pairs" is useful and citable.

### Hypotheses

| ID | Hypothesis | Role |
|---|---|---|
| **H1** | At a matched true-accept rate (TAR) on correct candidates, the best AR+diffusion pair has a lower joint false-accept rate (FAR) on incorrect candidates than the best cross-family AR+AR pair | **Primary** — the one confirmatory test |
| **H2** | Error overlap on wrong answers is lower across the AR/diffusion boundary than across AR families, after accounting for verifier competence | Mechanism — secondary |
| **E1** | Shared-lineage control: a diffusion model initialized from an AR model (Dream ← Qwen2.5-7B) behaves like a different family, or like its parent, when paired with that parent | Exploratory |

**Claims stay about the tested model pairs.** Off-the-shelf models differ in data, post-training and lineage as well as objective, so this study can show *whether* tested pairs are complementary, not *why* in a causal sense.

---

## 2. The frozen candidate bank

A fixed set of (problem, candidate answer, label) triples that every verifier judges identically. Everything downstream is conditional on identical inputs.

### Tasks (independently checkable only)

| Domain | Source | Label | Notes |
|---|---|---|---|
| Math | GSM8K (test labels from **GSM8K-Platinum**: 1,209 problems after 110 ambiguous ones were removed and 10 relabeled) and MATH (problems whose reference answer is a rational number, which both checkers read reliably): MATH-500, then the rest of the MATH test set | Final-answer match, cross-checked by two checkers (e.g. Math-Verify plus a numeric/sympy check); if they disagree, the candidate's matched pair is removed | GSM8K-Platinum fixes known label errors in the GSM8K test set |
| Code | HumanEval+ (164) and MBPP+ v0.2 (378), from EvalPlus | Pass all EvalPlus base and plus tests under EvalPlus's execution guard on a disposable cloud machine (`bank/label_evalplus.py`); each candidate run twice, flaky results excluded | EvalPlus adds 80× / 35× more tests than the originals, so fewer wrong programs are labeled correct |

Code is supply-limited: 390 of the 542 problems are left after the pilot (392 unused, of which two near-duplicates of development problems are dropped), about 850 candidates at the pilot's yield (est.).

**Math sources and mix.** The pilot used GSM8K-Platinum and MATH-500 in equal shares, and the gates were judged on that mix. The full bank keeps it: **half the math pairs from GSM8K-Platinum, half from MATH**. GSM8K-Platinum sets the size (1,059 unused problems, about 535 pairs at the pilot's 0.51 pairs per problem). MATH matches it, using the 239 unused MATH-500 problems first (about 195 pairs at 0.81 per problem) and then problems from the rest of the MATH test set (about 340 pairs, so generate for about 500 numeric-answer problems to leave a margin).

- **Draw rule.** The new MATH problems are a uniform seeded draw (seed 544) of 500 numeric-answer problems. After the 2-per-problem cap, let G be the number of GSM8K-Platinum pairs and M the number of MATH pairs. The math target is 2 × min(G, M), half from each. GSM8K pairs are taken in seeded random order; MATH pairs are taken MATH-500 first, then the new problems, each in seeded random order. The code target is every pair left after the 3-per-problem cap. Generator caps are shares of these targets. If a cap leaves a target unmet, the shortfall is reported, not filled. The choices this rule leaves open are fixed in the [record](decisions/2026-11-21-full-bank-details.md).
- **Where the new problems come from:** OpenAI's `prm800k` repository, `math_splits/train.jsonl`, keeping rows whose `unique_id` starts with `test/`. These are the MATH test problems outside MATH-500. Record the file's fields, its stated license and its hash at download; its terms are checked before the release (§9). (`hendrycks/competition_math` on Hugging Face is disabled after a takedown notice.)
- **Exclude** every MATH-500 problem, every pilot problem and every near-duplicate of one, checked by problem text as well as by id.
- **Why not a larger share of MATH:** in the pilot LLaDA's AUROC was 0.65 on GSM8K-Platinum and 0.55 on MATH-500 (about 125 answers each, wide intervals). Changing the mix after seeing that would change the test the gates were judged on, in either direction. So the mix is fixed here, before any new problem is generated, and results are reported per source.
- **Known limits, to report:** PRM800K trained on these MATH problems, so a verifier may have seen them. The question files are public with attribution; if the check before the release (§9) finds that the terms do not clearly allow redistributing MATH text, the release lists those problems by id only.

### Candidates

- **Generators:** 4 AR models of clearly different strength (3 at minimum), so the bank contains realistic errors rather than synthetic corruptions. They come from **families not used as verifiers** — no Qwen, Llama, Gemma, Mistral or their derivatives, and no SmolLM (Dream's fine-tuning data includes SmolLM2 data). The four used in the pilot and the full bank are OLMo-2-7B-Instruct (weak; it replaced Falcon3-1B), Phi-4-mini-instruct, Granite-3.3-8B-Instruct and GPT-4.1-mini (API).
- **Generation defaults** (checked on the development bank, then frozen with the bank): math uses a zero-shot "solve step by step and put the final answer in \boxed{}" prompt, temperature 0.7, top-p 0.95, 4 samples per problem per generator, at most 1,024 new tokens; code uses the EvalPlus standard prompt, temperature 0.8, 8 samples per problem per generator, at most 768 new tokens. Thinking modes are off. Every sample's generator, seed and settings are logged.
- **Matched pairs.** For each (problem, generator) cell that produced both a correct and a wrong answer, keep one of each, drawn with a fixed seed. This stops a verifier from scoring well just by recognizing hard problems or a weak generator's style. Limitation to report: only problems where a generator is sometimes right and sometimes wrong enter the bank, and the report states what fraction of cells qualified.
- **Filters:** no extractable answer, truncation, identity strings ("As an AI…"), near-duplicates within a problem, or a formatted verifier input plus that verifier's longest output (up to 32 answer slots plus 256 critique tokens, applied to every candidate because the critique subset is drawn after the bank is frozen) longer than 2,048 tokens for any verifier. That is Dream's documented input-plus-output context. A candidate removed for one verifier is removed for all, and if one member of a matched pair is removed, both are.
- **Caps:** at most 2 pairs per math problem and 3 per code problem, each from a different generator; no generator supplies more than 30% of a domain's pairs (40% if only three generators are available). A generator counts as available only if it can supply at least 10% of the domain's target. In pilot math GPT-4.1-mini could not (99% of its usable answers were correct), so math has three generators and a 40% cap.

### Labels

- Fully programmatic for the main bank, as in PROJECT.md: no main-bank label is set by hand. The audit measures label error, and a fix to the checker goes into a new pre-freeze bank version.
- **Audit:** on the pilot (for the data gate) and again on a fresh 200-candidate sample of the full bank before the bank freeze. Each audit takes 200 candidates stratified by label and task, judged blind by Codex, one at a time. One person then reads, blind to our label and to Codex's answer, every item where Codex disagrees with our label or gave no usable verdict, every GPT-written item, and 100 others drawn at random. Pass: at most 3 label errors in 200 (1.5%), which keeps the one-sided 95% upper bound below 4%; every error found counts. The audit can pass without reading all 200 only if there is no error among the 100 others, at most 3 errors were found in all, and the 95% bound, worked out over the items read in full and the others, is below 4%. In every other case the person reads all 200 and the 3-in-200 rule decides. A fail means fixing the checker and re-auditing a fresh sample. On the full bank the reader is A or B, who take no part in selection or threshold work, and F runs the commands, so the reader never holds the files that show Codex's answers. Details and the bound: [record](decisions/2026-11-21-label-check-codex-plus-one.md).
- **Error types** (secondary): code from execution logs (wrong output / timeout; EvalPlus's check counts exceptions as wrong output); math tagged by hand on a 200-item sample of wrong answers (arithmetic slip / misread problem / wrong method / formatting).
- **Known limits:** passing tests is an operational label, not a proof, and a correct final answer does not prove every reasoning step is correct.

### Size and splits

| Set | Size | Use |
|---|---|---|
| Development bank | ~150 candidates | Parsers and Yes/No token sets; the shared wording of T1–T3 (one wording per template, identical for every verifier); answer-slot lengths; Monte Carlo sample count; scoring-code checks. Never used to choose a template per verifier. Never reported as results. |
| Pilot | 500 candidates (250 math / 250 code) | Go/no-go gates (§7) and precision estimates. Exploratory. |
| Full bank | About 3,000 candidates: about 2,100 math and 850 code (est., §6), within PROJECT.md's 2,000–5,000 range. Drawn by the rule above | **Validation** (one third) for selection and thresholds; **test** (two thirds) for the final comparison |

- **Split by problem:** all candidates of a problem, and any near-duplicate problems, stay in one set. Full-bank problems exclude all development and pilot problems. The split is stratified by source.
- **Two freezes.** The **bank freeze** fixes the bank version and the validation/test split. The **preregistration freeze** (§5 step 5) fixes the selected pairs, thresholds and analysis code before test labels are joined.
- **Versions:** each bank version is immutable and identified by its hash. Corrections found before the preregistration freeze create a new version and trigger recomputation. After test labels are joined, the primary result stays on the preregistered bank version; results on any corrected version are reported only as a labeled sensitivity analysis.
- **Label custody:** labels are stored in a separate file per split, not in the shared bank file. For the full bank, `labeled.jsonl` and the test-label file are **not committed** until the test join (the pilot's were, because the pilot is exploratory). The same holds for code execution logs and math error tags, which reveal labels: A and B hand them to F with the test labels and delete their copies. **F performs the bank freeze** (commits the bank, split and hashes). The test-label file is held by the infrastructure lead (F, §11), outside the storage that roles C, D and E use, and is joined to scores only once, after the analysis manifest is committed. A and B hand the test-label file to F as soon as the bank version is built and split, before any validation analysis. They then delete their working copies and take no part in, and do not review, selection or threshold work. The full bank's audit and the math error tagging are done by A, B and F only, and the audit's files (except the sheet) stay with F until the join; the 200 tagged wrong answers come from the test split, so the test breakdown can use them, and are tagged before A and B delete their copies. Before the preregistration commit, no one joins test labels to scores or runs the checker on test rows outside the bank build, and F logs the single join. Because labels are programmatic, this is a team rule rather than a technical barrier, and the report says so.

---

## 3. Verifiers

All verifiers are 7–9B dense models (plus two small AR models), which fit a single 24 GB GPU in bf16 (measure peak memory in Phase 0).

| Role | Model | Notes |
|---|---|---|
| AR | Qwen2.5-7B-Instruct | Apache-2.0. Shares Dream's parent: Dream was initialized from the Qwen2.5-7B base model (E1). |
| AR | Llama-3.1-8B-Instruct | Gated, Llama 3.1 license (request access in Phase 0) |
| AR | Gemma-2-9B-it | Gated, Gemma license (request access in Phase 0). 9.24B parameters, about 18.5 GB in bf16, so tight on a 24 GB GPU (measure). |
| AR | Mistral-7B-Instruct-v0.3 | Apache-2.0 |
| Diffusion | LLaDA-8B-Instruct | MIT. Trained from scratch. Mask id 126336. Official code pins `transformers==4.38.2`. |
| Excluded (diffusion) | iLLaDA-8B-Instruct | **Excluded** (2026-10-25): median Yes+No mass 0.284 with thinking off, below the usable-scores gate. Apache-2.0. Released June 24, 2026; trained from scratch; runs on LLaDA's code with `mask_id=5`, but needs its own environment (§9). Much stronger than LLaDA (model card: GSM8K 89.0 vs 77.5, MATH 56.7 vs 42.2). Its official evaluations run at batch size 1, so it is scored unbatched. |
| Diffusion | Dream-v0-Instruct-7B | Apache-2.0. **Initialized from Qwen2.5-7B**, then 580B tokens of diffusion training. Pins `transformers==4.46.2`, `torch==2.5.1`; needs ≥ 20 GB. Its logits are shifted right by one position. Mask id 151666. |
| Small AR reference | Llama-3.2-3B-Instruct, Qwen2.5-1.5B-Instruct | Llama-3.2-3B is gate A's "small AR verifier" (§7). Qwen2.5-1.5B only widens H2's competence range. Llama-3.2 is gated (request access in Phase 0). Not in the primary comparison. |
| Strong-baseline check | Qwen3-8B (thinking off) | Secondary only: answers "the AR baseline is outdated" without breaking competence matching (§5). |

**Why these models.** They are the proposal's choices, kept on purpose: the AR models are of the same generation and size as the diffusion models they are compared with, so competence is roughly matched — which is exactly what keeps "a weak verifier passing off noise as diversity" (the proposal's main risk) under control. They all fit one 24 GB GPU and run on stable library versions. iLLaDA was added because it is far stronger than LLaDA at almost no extra cost; it was later excluded (above).

### Which models are in H1

The final test is fixed in §5. Its models come from the pilot gates (§7) and nothing after the pilot changes them.

| | In H1 (math) | Not in H1 |
|---|---|---|
| **AR** | Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B | Mistral-7B (fails gate A on the pilot; scored for H2's trend). Llama-3.2-3B and Qwen2.5-1.5B (gate A's reference and H2's trend). Qwen3-8B, unless the rule below applies |
| **Diffusion** | LLaDA-8B | Dream-7B (fails gate A in math, 0.592; used for E1 and reported on code). iLLaDA (excluded) |
| **Domain** | Math | Code: built, scored and reported descriptively, never in H1 |

- **How a verifier got in:** it passes the usable-scores gate, gate A and gate B in math on the pilot (§7). Dream × Qwen2.5 is never an H1 pair (shared initialization; E1).
- **If an AR verifier drops out at the go/no-go (§7):** with fewer than three left, the AR+AR baseline is the best pair among those left plus Qwen3-8B, which is never paired with Qwen2.5-7B (same family). Qwen3-8B joins the baseline without the pilot gates or test 4, as a stronger comparison, but must pass the reference and repeat-run checks (§4 tests 2–3). Its tokenizer is included in the bank's length filter from the start, in case it is needed. If no cross-family AR+AR pair can be formed, H1 is not run and the AR-family study is reported (§7).
- **There is at most one H1 test,** and which one is settled at the go/no-go.

---

## 4. How verifier scores are obtained

### Primary interface: direct verdict, read as a probability

Every verifier sees the same problem and candidate in its own chat template, with the same question, and answers Yes or No. The score is the model's own confidence: **s = log P(Yes) − log P(No)**, read in **one forward pass**.

- **AR models:** the next-token distribution at the start of the answer.
- **LLaDA / iLLaDA:** the prompt followed by masked answer slots; read the distribution at the first masked slot (one forward pass with full bidirectional attention).
- **Dream:** the same, but read one position earlier, because Dream's official code shifts its logits right by one.
- **Yes/No sets** use only spellings that are a single token in that model's tokenizer ("Yes", " Yes", "yes"…); a spelling that splits into several tokens is dropped. The total probability on Yes plus No is logged as a format check.
- The number of masked answer slots (1, 8 or 32) is chosen per diffusion model on the **development bank**: the one with the highest median Yes+No probability, with ties going to fewer slots. (AUROC on 150 items is too noisy to choose by.)

This keeps PROJECT.md's "accept/reject plus confidence" interface while giving fine-grained scores for thresholds, and it makes the cost per candidate the same kind of operation for AR and diffusion models.

### Prompt formats (for the stability gate)

Three wordings, used for every verifier: **T1** minimal ("Is the final answer correct? Answer Yes or No."), **T2** grader persona, **T3** inverted ("Is the final answer incorrect?", score sign flipped, to catch a "yes" bias). Exact wording: `harness/prompts.py`.

- **The primary template is T1 for every verifier**, as in PROJECT.md ("identical prompt template"). The development bank is used only for parser and token-set checks, the shared T1–T3 wording, answer-slot lengths and the Monte Carlo sample count, never to pick a template per verifier.
- All three wordings were run on the pilot. Gate B uses T1 and T2 (§7); T3 is a reported diagnostic and is not run on the full bank. T1 and T2 are run on the full bank, T2 for the same-model pseudo-pairs (§5). LLaDA's T2 pilot scores have a low Yes+No mass (0.54 math, 0.34 code); the usable-scores gate is defined on T1, so this is reported next to any T2 result.

### Secondary interfaces (ablations, cut first if behind)

| Interface | Definition | Purpose |
|---|---|---|
| **Reconstruction score** (PROJECT.md's ablation) | Masked-reconstruction likelihood of the candidate given the problem. Diffusion: LLaDA's `get_log_likelihood` Monte Carlo estimate, ported to Dream with its shift; the number of Monte Carlo samples (default 128) is set on the development bank as the smallest that gives two independent estimates a rank correlation ≥ 0.95 (about 32 expected). AR: exact log-probability, as a comparison arm. Reported per token, as AUROC, and as FAR at thresholds set on the 500 validation candidates. Runs on the pilot and on 500 validation and 500 test candidates of the math bank (reduced from 1,000, §9; on the full math bank only if used as the repair, §7). The AR arm covers the four AR models. | Checks whether verdict prompting under-uses the diffusion objective |
| **Answer-only control** | Math only (in code the program is the answer). Show the problem and the final answer without the reasoning, with T1, to the four AR models and the diffusion models that pass the gates, on the pilot and 1,000 test candidates. Reported as AUROC. | High scores here mean the verifier judges by re-solving or recall rather than by checking the reasoning (this control cannot tell those two apart) |
| **Critique, then verdict** | Up to 256 tokens of critique, then the verdict read as above. Covers the four AR models and the diffusion models that pass the gates, on 1,000 test candidates. Diffusion models use 256 denoising steps for 256 tokens (the official best setting), or 64 steps with block decoding if the budget requires; the setting is recorded. Reported as AUROC and cost only, with no threshold. | Checks whether the one-pass format disadvantages diffusion models. Not cost-matched, so never primary. |

### Scoring-code tests (for verdict scores, run before a verifier's scores are used in the primary comparison)

1. **Sanity set (smoke test, not a gate):** 50 hand-written items (25 obviously right, 25 obviously wrong) outside the bank. A low AUROC prompts a check for a bug (wrong mask id, missing shift, wrong template, padding leak). It no longer excludes a verifier; gate A on the pilot decides competence. Versions before 1.3 required AUROC ≥ 0.90, which only Qwen2.5-7B met (0.92; the other eight verifiers 0.73–0.89). This change was made after seeing those scores and is reported as post hoc.
2. **Reference parity:** on 20 prompts, run in each model's official environment, the custom scorer's distribution at the first answer slot matches the official model's own forward pass on the same masked input (for Dream, with the official sampler's shift applied): same top token, and Yes/No score within 1e-3. This compares one forward pass with one forward pass, which also holds with 8 or 32 answer slots, where the official sampler's final token can differ. For Dream this is also the shift check.
3. **Invariance:** repeat runs agree (Spearman > 0.999). Batched and unbatched scores must agree (|Δs| < 0.05) for any verifier scored in batches; every verifier is scored at batch size 1, because batching moved LLaDA and Dream by up to 0.25.
4. **Answer-swap sensitivity** (on 200 pilot items): replacing a correct final answer with a wrong one strictly lowers the score on ≥ 80% of items (a tie counts as not lowered), judged per domain on T1. The items are 100 correct math and 100 correct code pilot candidates (seeded draw, seed 544, at most one per problem where possible). Math: only the content of the last `\boxed{}` is replaced, with the final answer of the item's matched wrong partner. Code: the matched wrong partner is shown instead, as "a program for the same problem that fails the tests".

If a verifier fails test 4 after C and D have signed off that its code matches the official reference (test 2), it is treated as weak, a gate A failure in that domain, rather than as an open bug. Only math affects H1. Expect a failure in code: there test 4 is the same as ranking a matched pair, and on the pilot the best verifier does that for 72% of pairs (Qwen; the others 50–69%). Math is the open question, so test 4 runs first, as part of the go/no-go (§7). Tests 2 and 3 have passed for LLaDA and Dream, and the AR reference check for every AR model but Mistral (17 of 20). **Still to run: test 4 for every verifier, and a recorded repeat-run check for the AR models.**
5. **Environment check (logged, not gating):** each diffusion model's official sampler on 100 GSM8K problems, with its binomial confidence interval compared against the model card. A mismatch prompts an investigation, but a model is never dropped on this alone (at 100 items a correct setup misses a ±7-point band about 9% of the time for LLaDA).

---

## 5. Comparison procedure (follows the instructor feedback)

**All selection and every threshold are set on validation data and frozen before the test set is touched.**

### The final test (fixed)

| | |
|---|---|
| **Question** | At the same acceptance of correct answers, does the best AR+diffusion pair accept fewer wrong answers than the best AR+AR pair? (H1) |
| **Data** | Math, test split of the full bank (§2), used once |
| **Scores** | T1 verdict scores, for every verifier |
| **AR+diffusion pairs** | LLaDA+Qwen2.5, LLaDA+Llama-3.1, LLaDA+Gemma-2 |
| **AR+AR pairs** | Qwen2.5+Llama-3.1, Qwen2.5+Gemma-2, Llama-3.1+Gemma-2 |
| **Choosing** | In each group, the pair with the lowest validation FAR at 90% pair TAR. Thresholds come from validation and are frozen with it |
| **Measure** | ΔFAR = FAR(chosen AR+diffusion pair) − FAR(chosen AR+AR pair) on test, with the matching check and the problem-level bootstrap below |
| **Verdict** | The decision rule below, first row that applies |

Only the go/no-go in §7 can change this table, once, before any full-bank answer is scored, and only in the ways listed there. **No other reading of any rule changes the verdict.** The readings not adopted (gate B over T1–T3, the 0.90 sanity bar, the highest-AUROC dynamic range), code, Dream, Mistral, and the reconstruction score (unless the go/no-go selects it) are reported as background.

### Single-verifier metrics (PROJECT.md's metrics table)

For every verifier, per domain and pooled: **ROC and AUROC** (discrimination); **calibration** of P(Yes) — expected calibration error and reliability curves (descriptive, since the primary metrics are rank-based); and **GPU-seconds per candidate** (measured cost).

### On validation

1. **Pair rule (AND):** a pair accepts a candidate only if both verifiers accept it. `pair TAR = P(both accept | correct)`, `pair FAR = P(both accept | wrong)`.
2. **Thresholds:** as in PROJECT.md, each verifier is thresholded to accept the same share of correct answers, chosen so that **the pair accepts 90% of correct validation answers**. (Matching each verifier's own TAR is not enough on its own: two independent verifiers that each accept 90% jointly accept about 81%.) The rule is `analysis.metrics.pair_thresholds`. Thresholds are stored in each verifier's raw score units, per domain.
   - **Every other threshold comes from validation too,** by the same procedure: each single verifier at 90% TAR; every pair analyzed anywhere in this plan (the final test's pairs, E1 pairs, the cross-family pairs that include a small AR model for the H2 trend, same-model pseudo-pairs and Qwen3 pairs) at 80%, 90% and 95% pair TAR; and any template-averaged or budget-matched score. All are stored in the analysis manifest. The only test-tuned numbers are in the row labeled "ideal thresholds (test-tuned)", and no verdict in §5 or §8 uses it.
3. **Selection:** among the pairs in the final test, the AR+AR pair and the AR+diffusion pair with the lowest validation FAR on math are chosen. Ties go to the pair with the higher mean validation AUROC of its two members, then to alphabetical order of the model names. How often each pair wins across validation bootstrap resamples is reported (selection stability).
4. **Cost check:** measured GPU-seconds per candidate for both selected pairs, on the same GPU type, batch size 1 for every verifier. On the pilot's math answers (A100) every combination of pairs in the final test is within 1.5× (largest ratio 1.45), so the costs count as comparable and H1 uses T1 scores as they are. If the full-bank ratio for the selected pairs is above 1.5×, H1 is reported without the "comparable cost" claim, and the comparison with the cheaper pair averaging T1 and T2 is reported as secondary. H1 itself does not change.
5. **Preregistration freeze:** the selected pairs, thresholds, answer-slot lengths and the analysis code are committed with a hash before test labels are joined.

### On test (run once)

6. Apply the frozen thresholds and report each pair's **TAR and FAR with counts and 95% confidence intervals**, and **ΔFAR = FAR(AR+diffusion) − FAR(AR+AR)** (negative favors the mixed pair).
7. **Matching check:** because thresholds are frozen, test TAR will drift from 90%. The comparison counts as matched only if each pair's test TAR is within ±5 points of 90% and the two pairs are within 3 points of each other (point estimates; intervals reported; **post hoc**, ±3 until version 1.4, and the old rule's outcome is reported beside it). Otherwise the verdict is the decision rule's first row. Nothing is retuned on test labels.
8. **Uncertainty:** paired bootstrap that resamples **whole problems** (keeping all candidates of a problem together; for H1, the math test problems), 10,000 replicates, seed 544, percentile intervals, with thresholds held fixed. A secondary analysis repeats the validation-stage selection inside the bootstrap to show how much the choice of pairs itself varies.
9. **Pooling (secondary results only):** where math and code are combined, each domain is weighted by its number of wrong test answers, which is fixed by the frozen bank before unblinding. H1 is math only, so it is not pooled.

### Decision rule (fixed in advance)

Smallest effect worth caring about: **3 points of FAR** at the matched operating point. Rows are checked in order; the first that applies is the verdict.

| 95% CI for ΔFAR (test) | Verdict |
|---|---|
| Matching check failed (step 7) | Inconclusive (not matched); trade-off curves shown |
| Entirely below 0, point estimate ≤ −3 points | AR+diffusion pair better, by a meaningful margin |
| Entirely below 0, point estimate > −3 points | Better, but by less than the meaningful margin |
| 90% CI inside (−3, +3) points | No meaningful benefit (equivalence) |
| Entirely above 0 | AR+diffusion pair worse |
| Otherwise | Inconclusive; report the interval and the smallest effect the data could have detected |

### Reference rows in the same table

- The best single verifier at its validation threshold for 90% TAR (does any pair beat one good verifier?).
- **Same-model pseudo-pairs:** one verifier with templates T1 and T2, fused the same way. A cross-model pair's gain counts as diversity only if it beats the pseudo-pair of its stronger member (by validation AUROC) at the same 90% pair TAR: lower test FAR, with the two matched on test by step 7's limits. It reuses scores already computed, so it costs nothing.
- **Ideal thresholds (test-tuned):** the same comparison with thresholds tuned on test, to show how much frozen thresholds cost. Secondary; no verdict uses it.
- Qwen3-8B paired with the AR and diffusion partners that give the lowest validation FAR (strong-baseline check).

### Secondary results (descriptive)

- **Accuracy-versus-cost curves** (PROJECT.md's ensemble gain curves): FAR at the matched operating point against measured GPU-seconds, for single verifiers and all pairs (no ensembles larger than two). The critique-then-verdict variants appear as AUROC against cost, since they have no validation thresholds.
- The 80% and 95% TAR operating points, and all pairs in the final test, not just the selected two.
- **Breakdowns** by task, generator and error type, as in PROJECT.md. Small groups are descriptive only.

### H2 — mechanism

On wrong test answers, for every pair in the final test plus the E1 and H2-trend pairs, at the pair's frozen 90%-pair-TAR thresholds (step 2):
- **κ on false accepts** (chance-adjusted agreement on which wrong answers get accepted), with bootstrap intervals;
- Spearman correlation of scores on wrong answers (no threshold needed);
- **CAPA** (Goel et al., 2025) from the verdict probabilities, implemented from the paper and unit-tested; secondary, because it depends on calibration, which differs between model classes.

**Competence check:** across cross-family AR–AR pairs (including the two small AR models, so competence varies), fit κ against the pair's mean AUROC. AR–diffusion pairs count as "more diverse" only if they sit below that trend, with an interval excluding zero. Low overlap with a weak verifier alone is not evidence. A pair's status, from κ minus the trend, first that applies: below (95% interval under 0), above (over 0), on the trend (90% interval inside ±0.10 κ), inconclusive; 0.10 is a chosen number ([record](decisions/2026-11-21-mechanism-wording.md)). Reported by task and error type as well.

### E1 — shared-lineage control

Compare Qwen2.5-7B + Dream with Qwen2.5-7B + LLaDA and with Qwen2.5-7B + each AR model, on FAR and κ. If Qwen+Dream is as decorrelated as Qwen+LLaDA, the difference survives shared initialization; if it looks like an AR pair, lineage or data likely matters. Not causal: Dream also has 580B extra training tokens and its own fine-tuning data. E1 runs if Dream passes the usable-scores gate; if Dream fails gate A, E1 is still reported, but descriptively only.

---

## 6. Sample size

- **Pilot precision check (est., to be recomputed with committed code by E before the bank freeze):** on the pilot's 125 wrong math answers, each AR+diffusion pair and each AR+AR pair of the final test disagree on 8–23%, and answers to one problem cluster little (design effect about 1.0–1.2). About 2,100 math candidates give about 710 wrong test answers, so the detectable ΔFAR (80% power, 5% two-sided) is **about 3–5.5 points**, depending on which pairs are selected.
- **Method:** from pilot data, estimate how often the two candidate pairs disagree on wrong answers and how much answers to the same problem cluster. Then compute the smallest ΔFAR the planned test set can detect.
- **Earlier planning figures (est., made before the pilot and replaced by the check above):** a 5,000-candidate bank gives about 1,650 wrong test answers. With the pooling in §5 step 9, the detectable effect is then roughly 3.2–4.6 points, depending on how often the pairs disagree. So even the top of PROJECT.md's range may not reach the 3-point margin, and the "no meaningful benefit" verdict (90% interval inside ±3 points) is reachable only if the observed difference is near zero.
- **Decided (option a):** the bank stays within the approved range and the smallest detectable effect is reported alongside the verdict. Option (b), a bank above 5,000, is not used. Equal domain weights would make this worse (about 3.8–5.4 points), which is why §5 weights by wrong answers.
- **Code** (a few hundred wrong test answers) can detect only large effects, which is one reason it is descriptive (est.).

---

## 7. Pilot gates and decisions

Gate values were fixed on October 19, before the pilot was run on October 27. Some rules were changed or clarified afterwards; each is marked **post hoc** or **clarified** below and recorded in `decisions/`. They are project-management thresholds, not scientific standards. The template is fixed (T1) and answer-slot lengths are set on the development bank, so the pilot gates are not judged on the data used to tune them.

| Gate | Pass | If it fails |
|---|---|---|
| **Data and pipeline** | 500 labeled pilot candidates; split and provenance checks pass; audit within target; scoring-code tests 2 and 3 pass and test 4 has been run for every verifier in H1 except Qwen3-8B if it joins (test 1 is a smoke test, **post hoc**, §4; a test 4 failure is a gate A failure, §4). Mistral, not eligible, missed its AR reference check on 3 of 20 items by at most 0.125, a BF16 rounding step or two, and does not block this gate. Pilot AUROC, calibration (ECE) and runtime are reported per verifier, as the proposal's Milestone 1 promised. | Fix the pipeline before judging any model |
| **Usable scores** | Median Yes+No probability ≥ 0.5 under T1 for every verifier, checked on the development bank before the pilot (and reported on the pilot) | Switch the answer words to "Correct/Incorrect" for every verifier and every template, once, on the development bank, so the wording stays identical across verifiers. A verifier that still fails is excluded. |
| **Gate A — competence** (PROJECT.md; required of diffusion models, and the same bar decides which AR verifiers are in H1) | Per domain, on T1 scores: AUROC ≥ 0.60 with 95% lower bound > 0.50; diffusion models must also be **no more than 0.10 AUROC below the small AR verifier** (Llama-3.2-3B); being above it is fine (**clarified**). LLaDA's AUROC on validation is reported next to this bar, but does not change the test (**post hoc**, below). | Not in H1. If LLaDA fails, use the one repair below. |
| **Gate B — prompt stability** (PROJECT.md) | **AUROC spread across T1–T2 ≤ 0.05** for each verifier, per domain and pooled, on the pilot. **Post hoc:** the rule was T1–T3 until version 1.3. The T1–T3 outcome and the T3 results are reported beside it. | Investigate the scoring. Any verifier, AR or diffusion, that stays unstable is excluded from primary pairs. Templates are never picked on pilot or test data. |
| **Dynamic range** | The best single AR verifier's pilot FAR at 90% TAR is between 0.10 and 0.70 per domain (target 0.15–0.60). **Clarified:** "best" means the lowest FAR at 90% TAR, the same criterion §5 uses to select pairs. | **Closed.** It passed on the pilot and the bank's mix is fixed (§2), so there is no rebalancing; if labels or the pool change, the new value is only reported |
| **No shortcuts** | A length-only score and a surface-feature score (answer format, number of reasoning steps or lines of code, hedging phrases; fit on the development bank) each reach AUROC ≤ 0.65 on the pilot. Problem- and generator-level scores, including problem difficulty, are uninformative by construction, because every problem and generator supplies equal numbers of correct and wrong answers. | Tighten the matched-pair filters and re-check |
| **Compute** | Measured settings fit the confirmed GPU hours with 30% contingency | Cut in the order of §9 |

**Domain:** H1 is math only, and code is reported descriptively. (Until version 1.3 a code rule could admit code to H1, and H1 could narrow to code; both are removed.)

**Decisions:**
- **Continue** if LLaDA passes gates A and B and test 4 in math. Passing does not require beating AR+AR in the pilot.
- **Repair once** (time-boxed to one week of one person) if LLaDA does not pass: try the reconstruction score as LLaDA's score. If it passes, the H1 test is declared in the final model record and the preregistration as "AR verdict versus diffusion reconstruction score" and reported under that name — the interface change is never silent.
- **Pivot** if nothing passes after the repair: the **AR-family diversity study** on the same bank, harness and metrics — which AR families' mistakes differ most, and how close cross-family pairs come to a perfect verifier — plus the reported negative result "the tested diffusion models are not competent verdict verifiers on these tasks."
- Claims are restricted to math.

### Pilot outcome (October 27 pilot, 500 candidates)

| Rule | Outcome |
|---|---|
| **As written before the pilot (gate B over T1–T3)** | No diffusion model passes gates A and B in the same domain, so "repair". Gemma-2-9B is the only AR verifier that passes both gates in both domains. Under the old sanity bar as well, no AR verifier would be eligible: only Qwen passed that bar, and Qwen fails gate B over T1–T3. |
| **As adopted (gate B over T1–T2; sanity set as a smoke test)** | **Continue on math.** Diffusion: LLaDA. AR: Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B. |
| **Why T1–T2** | T3 gives every verifier, AR and diffusion, a lower pooled AUROC than T1, and most verifiers accept nearly everything under it (Qwen 98.8%, Llama-3.1 97.4%, LLaDA 99.8%, Dream 98.6%). Gemma-2 (62.8%) is the main exception. T1 and T2 agree within 0.05 for every verifier. Read this way, the T1–T3 spread mostly measures how models handle a negated question. This is an interpretation made after seeing the data, not a test of it. |
| **Code** | Under the old code rule code would have been admitted (three AR verifiers and Dream meet the bar there), but no diffusion model passes in both domains, so H1 narrowed to math. Version 1.3 fixes math. |
| **Dynamic range** | Math passes narrowly (Qwen 0.696; the highest-AUROC verifier, Gemma, would give 0.744 and fail). Code passes (Qwen 0.616). |
| **Not yet done** | The pilot label audit, test 4 and the AR repeat-run check. They complete the go/no-go below. |

### The go/no-go: made once, on the pilot, before any full-bank answer is scored

Three pilot checks are left: the label audit (§2), test 4 in math (§4), and a recorded repeat-run check for the AR models (§4 test 3; a failure there is a scorer bug to fix, not a verdict on the model). When they are done, the models and LLaDA's score are written to `decisions/` as the **final model record**. No full-bank answer is scored before that record is merged (building the banks can go ahead).

The verifiers that can be in H1 are LLaDA, Qwen2.5-7B, Llama-3.1-8B and Gemma-2-9B. The pool can only shrink: no other verifier enters, whatever the checks show (Qwen3-8B only as §3 says). Below, **fails** means failing gate A, gate B or test 4 in math on the pilot.

| What the pilot checks show | The final test |
|---|---|
| None of the four fails | **Go:** exactly the table in §5 |
| LLaDA fails, and its reconstruction score passes | **Repair:** the same table with LLaDA's reconstruction score in place of its verdict score (details below), reported as "AR verdict versus diffusion reconstruction score" and without the "comparable cost" claim |
| LLaDA fails, and its reconstruction score fails or is not finished within its one-week time box | **Pivot:** H1 is not run. The AR-family study is reported: the four AR verifiers, all six cross-family pairs, math and code, descriptively, with the negative result on diffusion |
| An AR verifier fails | It leaves the pool and §3's rule on a smaller pool applies. LLaDA is paired only with the AR verifiers left, never with Qwen3-8B |
| The audit finds more than 3 errors in 200 | Fix the checker, relabel the pilot, drop pairs that no longer hold one correct and one wrong answer, re-audit 200 fresh candidates, and recompute gates A and B and test 4 on the corrected labels. Then apply the rows above. If the second audit also fails, H1 is not run and the instructor is told |

Rows combine, and the result is recorded. Only the second row needs the reconstruction score before scoring starts, so D builds it in parallel and it delays nothing otherwise.

**Repair details.** Per-token masked-reconstruction log-likelihood with 128 Monte Carlo samples (§4's default), seed 544. It passes if it meets gate A on the pilot's math answers and a second estimate with seed 545 on the same answers has a rank correlation of at least 0.95 with it (this takes gate B's place, since the score has no wording). Test 4 is not applied to it.

**After the record is merged nothing changes the models, pairs, domain, score, threshold procedure or verdict rule.** In particular:
- **LLaDA's re-check is reported, not applied.** LLaDA passed gate A narrowly on the pilot (0.607 against 0.60), so on validation it is close to a coin flip whether its AUROC lands above 0.60. Letting that decide would pick the experiment after the bank is scored. If LLaDA is below the bar on validation (any of gate A's three parts fails), the H1 verdict is reported with that fact, and no claim about diversity is made from H2.
- **Why this is safe for H1:** the comparison is at the same acceptance of correct answers, so a weak verifier has to be given a loose threshold and its pair accepts more wrong answers. Weakness counts against the AR+diffusion pair.
- **Post hoc:** before version 1.3 a failed re-check dropped the model. The change is made before any validation data exists.

---

## 8. How each outcome is read

| H1 result | H2 result | Reading | Claim allowed |
|---|---|---|---|
| Better, meaningful | Below the AR trend | Tested diffusion verifiers add complementarity beyond strong cross-family AR pairs | "For the tested models, crossing the AR/diffusion boundary bought useful verification diversity" — correlational |
| Better, meaningful | On the AR trend | A competent, roughly independent extra verifier, but no more different than an equally competent AR model | Practical yes, mechanism no |
| No meaningful benefit | Below the AR trend | Different mistakes, but too weak to turn into fewer false accepts at 90% TAR | "Diversity without competence" — revisit as diffusion models improve |
| No meaningful benefit | On the AR trend | No useful diversity for these models, tasks and interface | Clean negative, scoped to the tested configurations |
| Worse | — | Adding the diffusion verifier as a veto hurts | Scoped advice: do not add these models as vetoes |
| Better, below the 3-point margin | Any | A real but small gain | "A small benefit for the tested pairs", with the interval |
| Inconclusive | — | Underpowered | Estimate and interval only |

Rows are read in order. An H2 result above the trend is read as "on the trend". If H2 is inconclusive (§5's status), or LLaDA is below gate A's bar on validation (§7), the H1 reading stands alone, with no mechanism claim.

E1 is read alongside: it can support or weaken a lineage explanation, but never proves one. Claims about H1 and H2 are about math; code results are descriptive.

---

## 9. Compute and reproducibility

- **Compute:** Colab Pro+ now; USC CARC allocation as the likely upgrade; modest API credits for generation. Every model fits one 24 GB GPU (measure in Phase 0).
- **Rough budget (est.):** candidate generation ≈ 10–20 A100-hours; primary scoring of ~10 models × ~5,000 candidates × 2 templates (plus 3 on the pilot) ≈ 5–15 A100-hours; reconstruction score on 3 diffusion models × ~2,000 candidates × ~32 Monte Carlo passes ≈ 5–15 A100-hours (about 2.5–3× more if all three diffusion models are repaired on the full bank); critique-then-verdict ≈ 10–20 more. Replace with Phase 0 measurements, and confirm Colab compute units and CARC status, since the total can exceed one Colab Pro+ subscription.
- **Measured on the pilot:** about 0.03–0.09 s per answer per model on an A100 (Dream 0.15 s on an L4), so scoring about 3,000 candidates with 8 models and 2 templates is roughly one GPU-hour plus model loading.
- **Environments:** one for LLaDA (`transformers 4.38.2`), one for iLLaDA (`transformers 4.57.1`, as pinned in its official evaluation script; its chat template ships as a `chat_template.jinja` file that 4.38.2 does not read), one for Dream (`transformers 4.46.2`, `torch 2.5.1`), and one for the AR models (`transformers ≥ 4.51` for Qwen3). Before scoring, each adapter saves the rendered prompt for one fixed example, and a test compares it with the model's official chat template. The shared boundary is the saved score file.
- **Measure, don't assume:** batch size, precision, token counts, steps, warm-up, synchronized time, peak memory and retries, on the same GPU type for cost comparisons. A100- and L4-seconds are never compared directly.
- **Saved contracts:**

| Artifact | Contents |
|---|---|
| Bank row | candidate id, pair id, problem id, source, domain, generator and settings, text and hash, checker version, split (no label) |
| Label file (one per split) | candidate id, label, checker outputs, error type (code from execution logs; math hand tags where sampled); the test file is held by F (§2) |
| Score row | bank hash, candidate id, model and revision, template, answer-slot length, score, P(Yes), P(No), Yes+No mass, status, runtime, memory |
| Run manifest | code revision, environment lock, hardware, command, timestamps, expected vs completed ids, output checksums |
| Analysis manifest | bank and split hashes, selected pairs, frozen thresholds, weights, bootstrap seed, output paths |

- **Repository layout:** `bank/`, `harness/`, `analysis/`, `results/`, `reports/`, plus `configs/`, `tests/`, `decisions/`. Large files live in shared storage; manifests live in the repository.
- **Tests:** labelers, split isolation, scoring code (§4), resume behavior, and the metrics on synthetic data with known answers. A second person reproduces the pilot and the final main table from saved outputs.
- **Release:** as in PROJECT.md, the labeled bank plus the overlap-analysis code is the project's public artifact, which survives any outcome. Generator and dataset terms are checked before the release (for example GSM8K-Platinum's added annotations are CC BY-SA 4.0); fields, stated licenses and hashes are recorded at download. Text or outputs whose terms do not clearly allow release are left out of it, and those rows are listed by id; the experiment does not change.

**Applied now, to finish as fast as possible:** (1)–(3) are not planned work; they are done only if someone is free after the main result is in, and the report says they were cut. The exception is Qwen3-8B, which C scores if the AR pool falls below three (§3). (4) is applied. (5) is not: code stays in, reported descriptively.

**Cut order if behind:** (1) critique-then-verdict ablation, (2) Qwen3 strong-baseline check, (3) answer-only control, (4) reconstruction ablation reduced to the pilot, 500 validation and 500 test candidates, (5) code domain. **Never cut:** the validation/test separation and freeze, the primary comparison, the audit, the same-model pseudo-pairs, gates A and B, and the generator and error-type breakdowns (a Milestone 2 commitment). Dropping the reconstruction ablation entirely, or the code domain, changes approved scope and is raised with the instructor first (§12).

---

## 10. Course deadlines and reporting

The remaining course deliverables. Internal work has no dates: every package in [NEXT_STEPS.md](NEXT_STEPS.md) is done as soon as its inputs exist.

| Deliverable | What | When |
|---|---|---|
| Final presentation | 6 minutes, 5 slides | Course window Nov 20 – Dec 3 |
| **Final report** | 8 pages + appendix, no late tokens | **Dec 15, 6:30 PM** |

- **Done:** midterm presentation; midterm report (version 1.3 settles the rules it left open, with one final test, §5, and one go/no-go before scoring, §7); the 500-candidate pilot (October 27).
- **Not done:** Milestone 2, the full comparison with generator and error-type breakdowns, planned for Nov 20. Tell the instructor it is late and that code is reported descriptively rather than in the primary result.
- **Minimum result:** H1 on math with the verifiers that pass the gates, with generator, source and error-type breakdowns, H2, E1 and the same-model pseudo-pairs.
- **If the presentation comes before the test result:** present the pilot, the settled rules and the frozen protocol, and label every pilot number in-sample.
- **What goes in the 8 pages:** question and design; bank and labels; verifiers and scoring; the primary H1 table and figure; H2 with the competence check; E1; limitations. **Appendix:** gates and pilot, full pair matrix, ablations, robustness, error types, prompts, reproducibility details.
- **Text reused from the proposal** gets the corrections recorded in the team's notes on the submitted version (the restored gate, grammar and punctuation).

---

## 11. Team roles (PROJECT.md's six roles)

Six packages, one per person, listed with their steps and hand-offs in [NEXT_STEPS.md](NEXT_STEPS.md). Each owner keeps their code, a methods paragraph and one report-ready result; a reviewer from another role approves merges.

| Role | Owns | Reviewer |
|---|---|---|
| **A — Bank: math** | GSM8K/MATH sources, generators, checkers, matched-pair selection, math error tags, label check (A or B reads, §2) | F |
| **B — Bank: code** | HumanEval+/MBPP+, EvalPlus execution, flaky-test reruns, code labels and error types, answer-swap items for test 4 | F |
| **C — Verifier harness** | Common scoring interface, prompt templates, AR adapters, Yes/No sets, batching and resume, cost logging | D |
| **D — Diffusion specialist** | LLaDA and Dream scoring, scoring-code tests, reconstruction score | C |
| **E — Metrics and statistics** | Validation selection and thresholds, the preregistration commit, bootstrap, decision rule, H2 and E1, power check, figures | C (A and B never review selection or thresholds, §2) |
| **F — Infrastructure and writing** | Environments, GPU access, run ledger, bank freeze, **test-label custody** and the logged label join, report and slide integration, release | B |

A short check-in whenever a hand-off in NEXT_STEPS.md completes. Design changes are recorded in `decisions/` and are possible only until the final model record is merged (§7). After that the final test (§5) does not change: models, pairs, domain, score, threshold procedure and verdict rule are fixed, and anything else is a labeled background analysis.

---

## 12. Risks

| Risk | Control |
|---|---|
| Diffusion verifiers too weak or badly calibrated | Gates A and B; one repair; declared AR-only pivot |
| Silent scoring bugs (mask ids, Dream's shift, padding with bidirectional attention, library versions) | Scoring-code tests (§4); separate environments; cross-review between C and D |
| Confounding from data, post-training or lineage | Claims about tested pairs only; E1 shared-lineage control; generators outside verifier families |
| Label noise | Programmatic labels, stronger code tests, GSM8K-Platinum, audit |
| Too little data to detect 3 points | About 3–5.5 points detectable (§6), reported with the verdict; "inconclusive" is a planned outcome (§5) |
| LLaDA fails test 4 in math | Decided before scoring by the table in §7: reconstruction score, then pivot |
| LLaDA is below the competence bar on validation | Reported; the test is unchanged and no mechanism claim is made (§7) |
| Rules changed after seeing the pilot | The post hoc rule changes (§13 rows 29–31, 33, 36–38) and the settled terms (rows 32, 42) recorded in `decisions/`, the original outcome reported beside the adopted one, and H1 tested only on new problems under frozen thresholds |
| New MATH problems seen in training, or not redistributable | Results by source; release by problem id if terms are unclear (§2) |
| Memorized benchmark answers | Exposure differs by model and is unknown; the answer-only control is run only if time allows (§9), and claims are scoped |
| Compute shortfall | Everything fits one 24 GB GPU; cut order (§9) |
| Changes the course did not approve | §13 lists them all. The midterm report names the ones that touch approved scope and asks the instructor to flag objections: four AR families instead of 2–3; three diffusion verifiers instead of 1–2, plus small AR references and a Qwen3 check; the validation/test split with a pilot kept separate from the full bank; a possible math-only primary result; and no causal claims. **New since the midterm:** gate B over T1–T2, math problems from the rest of the MATH test set, and Milestone 2 running late are told to the instructor; the sanity-set change was already in the midterm report. |

---

## 13. Changes from PROJECT.md

Each change has one of five sources: **Feedback** (instructor), **Review** (a flaw confirmed by independent reviews), **Release** (a model released since), **Course** (course logistics), or **Pilot** (made after seeing pilot results).

| # | PROJECT.md | This plan | Why | Source |
|---|---|---|---|---|
| 1 | Matched operating points; where they are set is not specified | Pairs selected and thresholds set on validation, frozen, then applied once to test, with a matching check | Keeps model selection separate from the evidence | Feedback |
| 2 | Thresholds equalize each verifier's TAR | Same, with the shared level chosen so the *pair* accepts 90% of correct answers | Equal individual TARs do not fix the pair's TAR | Review |
| 3 | Bootstrap over candidates | Bootstrap over whole problems | Answers to the same problem are not independent | Review |
| 4 | Bank of correct and incorrect candidates | Matched correct/wrong pairs within each (problem, generator) cell | Stops verifiers scoring well by spotting hard problems or a weak generator | Review |
| 5 | 3–4 AR generators of varied strength; families unspecified | 4 generators (3 minimum), from families not used as verifiers | Verifiers tend to favor their own family's text | Review |
| 6 | GSM8K/MATH-style; MBPP/HumanEval-style | GSM8K-Platinum labels; EvalPlus HumanEval+/MBPP+ tests | Fewer wrong labels (PROJECT.md risk 3) | Review |
| 7 | Accept/reject plus confidence | Confidence read from the model's Yes/No probabilities in one pass | Fine-grained scores for thresholds; the same kind of operation for AR and diffusion, so cost stays comparable | Review |
| 8 | 1–2 diffusion verifiers: LLaDA-8B, Dream-7B | LLaDA-8B and Dream-7B kept; **iLLaDA-8B added** (three in total) | Much stronger, from scratch, runs on LLaDA's code | Release |
| 9 | AR: 2–3 families (Qwen2.5, Llama-3.1, Gemma-2 or Mistral) | All four of those families | Six AR+AR pairs instead of three makes the baseline harder to beat; one extra 7B pass per answer. After the pilot gates, H1 uses three of them (§5) | Review |
| 10 | "Within striking distance of a small AR verifier" | Small AR verifiers named (Llama-3.2-3B, Qwen2.5-1.5B); gate A measured against Llama-3.2-3B | Makes gate A checkable; also gives H2 a competence range | Review |
| 11 | — | Qwen3-8B as a secondary strong-baseline check | Answers "the AR baseline is outdated" without breaking competence matching | Review |
| 12 | — | Same-model pseudo-pairs as a reference | Separates real diversity from prompt noise; free | Review |
| 13 | Negative result is informative | Fixed 3-point margin and a decision table including "no meaningful benefit" | Makes a negative result a testable outcome | Review |
| 14 | Record GPU-seconds | Measured cost must be within 1.5×, else the budget-matched comparison decides H1 (changed by row 33) | Makes "comparable inference cost" checkable | Review |
| 15 | Gate A (for diffusion verifiers) and gate B | Kept and made numeric. Gate A judged per domain on T1 and reported again on validation (row 36); its competence bar also decides which AR verifiers are in H1. Gate B judged per domain and pooled, for every verifier. Added scoring-code tests, usable-scores, dynamic-range, no-shortcut and compute gates; answer-slot lengths set on a separate development bank | Gates that cannot pass by chance (gate A's lower bound), hide a domain failure, or be judged on the data used to tune them | Review |
| 16 | Pivot to AR-family study | Kept; one declared repair (reconstruction score) first, never silent | The proposal's motivation for reconstruction scoring | Review |
| 17 | Shared-lineage control where feasible; any AR+diffusion pair | Qwen2.5-7B-Instruct + Dream as E1 (Dream is initialized from the Qwen2.5-7B base); Dream × Qwen2.5 is excluded from H1 | Available at no extra cost; a shared-lineage pair would blur the cross-boundary comparison | Review |
| 18 | CAPA / κ on mistakes | κ on false accepts and score correlation first, on wrong answers only (false-accept overlap); CAPA secondary; competence trend check | Joint false accepts are what H1 measures; CAPA depends on calibration, which differs between model classes | Review |
| 19 | 2,000–5,000 candidates | Same range; target set by §2's draw rule, with the detectable effect reported; growing beyond it would be raised with the instructor | The 3-point margin may need the top of the range | Review |
| 20 | Dated timeline; pilot covers both task types and all models | Course deadlines kept (§10); internal dates set by the team; a reduced pilot (math only; code and Dream follow) if the full one cannot finish by Oct 27 | Oct 12 pilot milestone has passed | Course |
| 21 | 500-item pilot grows into the 2,000–5,000 bank | Pilot and development bank kept separate from the full bank | The data used to set gates and tune lengths stays out of the final test | Review |
| 22 | Bank never modified mid-experiment | Corrections before the freeze create new versions; after unblinding, the primary result stays on the preregistered version | Same aim, made workable when a checker bug is found early | Review |
| 23 | Math and code both in the main comparison | H1 is math only and code is descriptive (until version 1.3, code could enter under a code rule and H1 could narrow to either domain); dropping code entirely is raised with the instructor | Avoids pooling a domain where the verifiers are not competent | Review |
| 24 | Weighting of math and code unspecified | Domains weighted by their number of wrong test answers, fixed before unblinding | Most precise pooled estimate; equal weights would need a larger bank | Review |
| 25 | If compute tightens, shrink the bank or model count | Cut order in §9, which trims ablations first and never drops the generator and error-type breakdowns | Protects the primary comparison and Milestone 2 commitments | Review |
| 26 | Causal claims allowed if the controls support them | No causal claims; E1 can support or weaken a lineage explanation only | The available control (Dream vs Qwen2.5) also differs in training data and fine-tuning, so it cannot isolate the objective | Review |
| 27 | API credits for any closed-model AR baselines (optional) | Not used; API credits go to one generator | Budget, and closed models cannot be scored reproducibly | Course |
| 28 | — | Added: answer-only control, critique-then-verdict ablation, label audit, development bank, generation defaults, AR arm of the reconstruction score, input-length and per-problem caps, the generator-share cap and quality/consistency filters, MATH limited to checkable answers, usable-scores and compute gates, label custody, a fallback if the AR pool shrinks, extra repository folders; accuracy-versus-cost curves cover single verifiers and pairs only | Each closes a gap a reviewer identified; none changes the question | Review |
| 29 | Gate B on prompt variants (PROJECT.md); T1–T3 in this plan until version 1.3 | Gate B over T1–T2; T3 is a reported diagnostic | T3 fails most verifiers of both classes, and under T1–T3 only one AR verifier is eligible in both domains. **Post hoc** | Pilot |
| 30 | — (sanity AUROC ≥ 0.90 was a gate in this plan until version 1.3) | Sanity set is a smoke test; gate A on the pilot decides competence | Only Qwen met the bar after the code was verified; the test is 50 items. **Post hoc**, disclosed in the midterm report | Pilot |
| 31 | GSM8K/MATH-style math (the pilot used GSM8K-Platinum and MATH-500) | Math bank adds numeric-answer problems from the rest of the MATH test set, keeping the pilot's even GSM8K/MATH mix | More math data while keeping the mix the gates were judged on. **Post hoc**, fixed before any new problem is generated | Pilot |
| 32 | Dynamic-range "best verifier" undefined; gate A's "within 0.10" ambiguous; test 4's items and scope, the reconstruction repair's scope and its replacement for gate B undefined | "Best" = lowest FAR at 90% TAR; gate A one-sided; test 4 built and judged per domain (§4); repair applies to LLaDA on math (§7) | Midterm feedback: rules must be settled before the test. "Best" matches §5's selection rule; the other readings are reported. Decided after the pilot | Feedback (midterm) |
| 33 | Record GPU-seconds (row 14: the budget-matched comparison decides H1 above 1.5×) | Costs are judged comparable from the pilot (every ratio at most 1.45); H1 always uses T1 scores; the budget-matched comparison is secondary | One final test | Feedback (midterm) |
| 34 | Cut order applied only if behind | Cuts (1)–(4) applied now; Qwen3-8B returns if the AR pool falls below three | Time left | Course |
| 35 | — | Brought up to date with the pilot: OLMo-2 replaces Falcon3 and GPT-4.1-mini is the API generator; the generator cap counts only generators that can supply 10% of a domain; EvalPlus's guard replaces the isolated container for code labels; iLLaDA excluded; every verifier scored at batch size 1; the AR models use SDPA attention, except Gemma-2 (eager) | Already done in the pilot; most of it disclosed in the midterm report | Pilot |
| 36 | Gate A re-checked on validation, dropping a model that fails (this plan until version 1.3) | Reported on validation, not applied; go/no-go made once on the pilot, before scoring | The test must be settled before the bank is scored. **Post hoc**, made before any validation data exists | Feedback (midterm) |
| 37 | Primary domain could be math, code or both | Math, fixed | One final test. **Post hoc** | Feedback (midterm) |
| 38 | Matching check: both pairs within 3 points of 90% and of each other (this plan until version 1.4) | Each pair within 5 points of 90%, the two within 3 points of each other | Frozen thresholds drift; a simulation on pilot scores called about 30% of comparisons unmatched under the old limits and about 15% under the new ones. The limit between the pairs stays, because that is what moves ΔFAR. **Post hoc**, before any validation data exists | Pilot |
| 39 | — (this plan until version 1.4: two people label the audit, a third resolves) | Codex judges all 200 blind; one person reads the disagreements, the GPT-written items and 100 others, and all 200 when that is not enough to pass | Time left; the pass mark is unchanged | Course |
| 40 | — (this plan until version 1.4: terms checked before generating) | Terms checked before the release; fields, stated licenses and hashes recorded at download | Only the release depends on the answer | Course |
| 41 | — (this plan until version 1.4: 392 code problems; three code error types; draw and split choices left open) | 390 code problems; two code error types; the open choices fixed; question files public with attribution | Found while building the bank, before any answer is generated | Pilot |
| 42 | — (this plan until version 1.4: "on the trend", "below gate A's bar", "beats the pseudo-pair" and the side experiment's domain undefined) | Defined (§4, §5, §7, §8) | One reading before the preregistration; none changes the H1 verdict. Decided after the pilot | Pilot |

**Considered and not adopted** (from an independently written alternative plan): newer AR models as the primary pool (breaks competence matching), DiffusionGemma and Gemma 4 26B (needs an 80 GB GPU), Nemotron-Labs-Diffusion, a five-model competence ladder, an 8,000-candidate bank, and an AceInstruct-based lineage contrast. They add scope the course format cannot hold. The alternative plans and their reviews are kept with the team's working files, outside this repository.

---

## 14. References (verify exact citations before each report)

- Saad-Falcon et al., 2025 — *Shrinking the Generation-Verification Gap with Weak Verifiers* (Weaver). Note: its "correlated biases on hard examples" is the authors' explanation for diminishing gains, not a measurement.
- Goel et al., ICML 2025 — *Great Models Think Alike and this Undermines AI Oversight* (CAPA).
- Nie et al., 2025 — *Large Language Diffusion Models* (LLaDA); iLLaDA model card and paper (2026); Ye et al., 2025 — Dream 7B.
- Panickssery et al., 2024 — *LLM Evaluators Recognize and Favor Their Own Generations* (generator choice).
- Closest prior work, diffusion models checking themselves or drafting (verify each before citing): PRISM, arXiv:2602.01842; DEER, arXiv:2512.15176; HIVE, arXiv:2604.26139; FreeCache, arXiv:2505.21467. PRISM and DEER are the closest and were missing from the midterm presentation.
- Task sources: GSM8K (Cobbe et al., 2021) and GSM8K-Platinum (Vendrow et al., 2025); MATH (Hendrycks et al., 2021); MBPP (Austin et al., 2021) and HumanEval (Chen et al., 2021), with EvalPlus (Liu et al., 2023).
