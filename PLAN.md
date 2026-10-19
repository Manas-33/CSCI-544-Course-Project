# Plan — Do Diffusion Language Models Provide Useful Verification Diversity?

**CSCI 544 (Applied NLP, USC, Fall 2026) · Team "project 16" · Research-focused · Version 1.2, October 19, 2026 (revised after two independent checks)**

This is the team's working plan. It is built from [PROJECT.md](PROJECT.md) (the project idea and design) and the execution rules of the earlier team plan, follows the instructor's feedback on the proposal (pairs and thresholds chosen on validation data and frozen before the test set; kept with the team's course records), and adds only improvements that independent reviews confirmed fix real problems. Every difference from PROJECT.md is listed in §13 with its reason. Internal phase dates are left to the team; course deadlines are fixed (§10).

Notation: **(est.)** marks a planning estimate to be replaced by a measurement. **Phase 0** is the setup work before the pilot (§10).

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
| Math | GSM8K (test labels from **GSM8K-Platinum**: 1,209 problems after 110 ambiguous ones were removed and 10 relabeled) and MATH (problems whose answers the checker handles reliably) | Final-answer match, cross-checked by two checkers (e.g. Math-Verify plus a numeric/sympy check); if they disagree, the candidate's matched pair is removed | GSM8K-Platinum fixes known label errors in the GSM8K test set |
| Code | HumanEval+ (164) and MBPP+ v0.2 (378), from EvalPlus | Pass all EvalPlus tests in an isolated container: no network, fixed time and memory limits; each candidate run twice, flaky results excluded | EvalPlus adds 80× / 35× more tests than the originals, so fewer wrong programs are labeled correct |

Code is supply-limited (542 problems), so it will hold fewer candidates than math: roughly 500–800 in the full bank (est.), once development and pilot problems are removed and only cells with both a correct and a wrong answer count. Measure the share of qualifying cells on the pilot before fixing the bank size.

### Candidates

- **Generators:** 4 AR models of clearly different strength (3 at minimum), so the bank contains realistic errors rather than synthetic corruptions. They come from **families not used as verifiers** — no Qwen, Llama, Gemma, Mistral or their derivatives, and no SmolLM (Dream's fine-tuning data includes SmolLM2 data). Candidates to confirm for access and license in Phase 0: Falcon3-1B-Instruct (weak), Phi-4-mini-instruct (mid), Granite-3.3-8B-Instruct (mid–strong), and one API model for subtle MATH errors. Falcon3 uses the TII Falcon-LLM License 2.0, not Apache-2.0: confirm now that its outputs can be used and released with the bank, or swap it for another weak model from an unused family.
- **Generation defaults** (checked on the development bank, then frozen with the bank): math uses a zero-shot "solve step by step and put the final answer in \boxed{}" prompt, temperature 0.7, top-p 0.95, 4 samples per problem per generator, at most 1,024 new tokens; code uses the EvalPlus standard prompt, temperature 0.8, 8 samples per problem per generator, at most 768 new tokens. Thinking modes are off. Every sample's generator, seed and settings are logged.
- **Matched pairs.** For each (problem, generator) cell that produced both a correct and a wrong answer, keep one of each, drawn with a fixed seed. This stops a verifier from scoring well just by recognizing hard problems or a weak generator's style. Limitation to report: only problems where a generator is sometimes right and sometimes wrong enter the bank, and the report states what fraction of cells qualified.
- **Filters:** no extractable answer, truncation, identity strings ("As an AI…"), near-duplicates within a problem, or a formatted verifier input plus that verifier's longest output (up to 32 answer slots plus 256 critique tokens, applied to every candidate because the critique subset is drawn after the bank is frozen) longer than 2,048 tokens for any verifier. That is Dream's documented input-plus-output context. A candidate removed for one verifier is removed for all, and if one member of a matched pair is removed, both are.
- **Caps:** at most 2 pairs per math problem and 3 per code problem, each from a different generator; no generator supplies more than 30% of pairs (40% if only three generators are available).

### Labels

- Fully programmatic for the main bank, as in PROJECT.md: no main-bank label is set by hand. The audit measures label error, and a fix to the checker goes into a new pre-freeze bank version.
- **Audit:** done by A, B and F, on the pilot (for the data gate) and again on a fresh 200-candidate sample of the full bank before the bank freeze. Each audit takes 200 candidates stratified by label and task, labeled independently by two people, with a third resolving disagreements. Pass: at most 3 label errors in 200 (1.5%), which keeps the one-sided 95% upper bound below 4%. Otherwise fix the checker and re-audit a fresh sample.
- **Error types** (secondary): code from execution logs (wrong output / exception / timeout); math tagged by hand on a 200-item sample of wrong answers (arithmetic slip / misread problem / wrong method / formatting).
- **Known limits:** passing tests is an operational label, not a proof, and a correct final answer does not prove every reasoning step is correct.

### Size and splits

| Set | Size | Use |
|---|---|---|
| Development bank | ~150 candidates | Parsers and Yes/No token sets; the shared wording of T1–T3 (one wording per template, identical for every verifier); answer-slot lengths; Monte Carlo sample count; scoring-code checks. Never used to choose a template per verifier. Never reported as results. |
| Pilot | 500 candidates (250 math / 250 code) | Go/no-go gates (§7) and precision estimates. Exploratory. |
| Full bank | Within PROJECT.md's 2,000–5,000 range; target set after the pilot (§6) | **Validation** (one third) for selection and thresholds; **test** (two thirds) for the final comparison |

- **Split by problem:** all candidates of a problem, and any near-duplicate problems, stay in one set. Full-bank problems exclude all development and pilot problems.
- **Two freezes.** The **bank freeze** fixes the bank version and the validation/test split. The **preregistration freeze** (§5 step 5) fixes the selected pairs, thresholds and analysis code before test labels are joined.
- **Versions:** each bank version is immutable and identified by its hash. Corrections found before the preregistration freeze create a new version and trigger recomputation. After test labels are joined, the primary result stays on the preregistered bank version; results on any corrected version are reported only as a labeled sensitivity analysis.
- **Label custody:** labels are stored in a separate file per split, not in the shared bank file. The test-label file is held by the infrastructure lead (F, §11), outside the storage that roles C, D and E use, and is joined to scores only once, after the analysis manifest is committed. A and B hand the test-label file to F as soon as the bank version is built and split, before any validation analysis. They then delete their working copies and take no part in, and do not review, selection or threshold work. The audit and the math error tagging are done by A, B and F only. Before the preregistration commit, no one joins test labels to scores or runs the checker on test rows outside the bank build, and F logs the single join. Because labels are programmatic, this is a team rule rather than a technical barrier, and the report says so.

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
| Diffusion | iLLaDA-8B-Instruct | Apache-2.0. Released June 24, 2026; trained from scratch; runs on LLaDA's code with `mask_id=5`, but needs its own environment (§9). Much stronger than LLaDA (model card: GSM8K 89.0 vs 77.5, MATH 56.7 vs 42.2). Its official evaluations run at batch size 1, so it is scored unbatched. |
| Diffusion | Dream-v0-Instruct-7B | Apache-2.0. **Initialized from Qwen2.5-7B**, then 580B tokens of diffusion training. Pins `transformers==4.46.2`, `torch==2.5.1`; needs ≥ 20 GB. Its logits are shifted right by one position. Mask id 151666. |
| Small AR reference | Llama-3.2-3B-Instruct, Qwen2.5-1.5B-Instruct | Llama-3.2-3B is gate A's "small AR verifier" (§7). Qwen2.5-1.5B only widens H2's competence range. Llama-3.2 is gated (request access in Phase 0). Not in the primary comparison. |
| Strong-baseline check | Qwen3-8B (thinking off) | Secondary only: answers "the AR baseline is outdated" without breaking competence matching (§5). |

**Why these models.** They are the proposal's choices, kept on purpose: the AR models are of the same generation and size as the diffusion models they are compared with, so competence is roughly matched — which is exactly what keeps "a weak verifier passing off noise as diversity" (the proposal's main risk) under control. They all fit one 24 GB GPU and run on stable library versions. iLLaDA is added because it is far stronger than LLaDA at almost no extra cost.

### Pair eligibility for H1

- **AR+AR:** the 6 cross-family pairs among the four AR models.
- **AR+diffusion:** each diffusion model that passes the gates, paired with each AR model, **except Dream × Qwen2.5** (shared initialization; used for E1 instead).
- Small AR models and Qwen3-8B are never in the primary pairs.
- **Domains:** the primary result covers math, plus code if the §7 code rule admits it. The competence bar (gate A's AUROC criteria) applies to every primary verifier, AR or diffusion, per domain. A pair is eligible only if both members pass the usable-scores gate, the competence bar and gate B in every domain the primary result covers. Selection, thresholds and H1 then use only eligible pairs, with FAR pooled using the §5 step 9 weights; per-domain results for all pairs are secondary. If no diffusion model is eligible in every covered domain, the primary result is narrowed to one domain (math if a diffusion model is eligible there, otherwise code) and the other domain is reported descriptively. **There is always exactly one primary H1 test.**
- **If the AR pool shrinks:** if fewer than three AR verifiers stay eligible, the AR+AR baseline is the best pair among the eligible ones plus Qwen3-8B. This is recorded as a deviation and reported.

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

Three wordings, used for every verifier: **T1** minimal ("Is the final answer correct? Answer Yes or No."), **T2** grader persona, **T3** inverted ("Does this solution contain an error…?", score sign flipped, to catch a "yes" bias).

- **The primary template is T1 for every verifier**, as in PROJECT.md ("identical prompt template"). The development bank is used only for parser and token-set checks, the shared T1–T3 wording, answer-slot lengths and the Monte Carlo sample count, never to pick a template per verifier.
- All three wordings are run on the full pilot for gate B. T1 and T2 are also run on the full bank, for the same-model pseudo-pairs (§5).

### Secondary interfaces (ablations, cut first if behind)

| Interface | Definition | Purpose |
|---|---|---|
| **Reconstruction score** (PROJECT.md's ablation) | Masked-reconstruction likelihood of the candidate given the problem. Diffusion: LLaDA's `get_log_likelihood` Monte Carlo estimate, ported to Dream with its shift; the number of Monte Carlo samples (default 128) is set on the development bank as the smallest that gives two independent estimates a rank correlation ≥ 0.95 (about 32 expected). AR: exact log-probability, as a comparison arm. Reported per token, as AUROC, and as FAR at thresholds set on the 500 validation candidates. Runs on the pilot, 500 validation and 1,000 test candidates (and on the full bank only if used as the repair, §7). | Checks whether verdict prompting under-uses the diffusion objective |
| **Answer-only control** | Math only (in code the program is the answer). Show the problem and the final answer without the reasoning, with T1, to the four AR models and the diffusion models that pass the gates, on the pilot and 1,000 test candidates. Reported as AUROC. | High scores here mean the verifier judges by re-solving or recall rather than by checking the reasoning (this control cannot tell those two apart) |
| **Critique, then verdict** | Up to 256 tokens of critique, then the verdict read as above. Covers the four AR models and the diffusion models that pass the gates, on 1,000 test candidates. Diffusion models use 256 denoising steps for 256 tokens (the official best setting), or 64 steps with block decoding if the budget requires; the setting is recorded. Reported as AUROC and cost only, with no threshold. | Checks whether the one-pass format disadvantages diffusion models. Not cost-matched, so never primary. |

### Scoring-code tests (must pass before any score is used)

1. **Sanity set:** 50 hand-written items (25 obviously right, 25 obviously wrong) outside the bank; every verifier reaches AUROC ≥ 0.90. A lower value almost always means a bug (wrong mask id, missing shift, wrong template, padding leak).
2. **Reference parity:** on 20 prompts, run in each model's official environment, the custom scorer's distribution at the first answer slot matches the official model's own forward pass on the same masked input (for Dream, with the official sampler's shift applied): same top token, and Yes/No score within 1e-3. This compares one forward pass with one forward pass, which also holds with 8 or 32 answer slots, where the official sampler's final token can differ. For Dream this is also the shift check.
3. **Invariance:** batched and unbatched scores agree (|Δs| < 0.05); repeat runs agree (Spearman > 0.999). iLLaDA is scored unbatched, so for it this checks repeat runs only.
4. **Answer-swap sensitivity** (on 200 pilot items): replacing a correct final answer with a wrong one lowers the score on ≥ 80% of items. For code, the swap is a program for the same problem that fails the tests.

If a verifier fails test 1 or test 4 after C and D have signed off that its code matches the official reference (test 2), it is treated as weak, a gate A failure, rather than as an open bug.
5. **Environment check (logged, not gating):** each diffusion model's official sampler on 100 GSM8K problems, with its binomial confidence interval compared against the model card. A mismatch prompts an investigation, but a model is never dropped on this alone (at 100 items a correct setup misses a ±7-point band about 9% of the time for LLaDA).

---

## 5. Comparison procedure (follows the instructor feedback)

**All selection and every threshold are set on validation data and frozen before the test set is touched.**

### Single-verifier metrics (PROJECT.md's metrics table)

For every verifier, per domain and pooled: **ROC and AUROC** (discrimination); **calibration** of P(Yes) — expected calibration error and reliability curves (descriptive, since the primary metrics are rank-based); and **GPU-seconds per candidate** (measured cost).

### On validation

1. **Pair rule (AND):** a pair accepts a candidate only if both verifiers accept it. `pair TAR = P(both accept | correct)`, `pair FAR = P(both accept | wrong)`.
2. **Thresholds:** as in PROJECT.md, each verifier is thresholded to accept the same share of correct answers, chosen so that **the pair accepts 90% of correct validation answers**. (Matching each verifier's own TAR is not enough on its own: two independent verifiers that each accept 90% jointly accept about 81%.) Thresholds are stored in each verifier's raw score units, per domain.
   - **Every other threshold comes from validation too,** by the same procedure: each single verifier at 90% TAR; every pair analyzed anywhere in this plan (eligible pairs, E1 pairs, the cross-family pairs that include a small AR model for the H2 trend, same-model pseudo-pairs and Qwen3 pairs) at 80%, 90% and 95% pair TAR; and any template-averaged or budget-matched score. All are stored in the analysis manifest. The only test-tuned numbers are in the row labeled "ideal thresholds (test-tuned)", and no verdict in §5 or §8 uses it.
3. **Selection:** among pairs still eligible after the validation re-check of gate A (§7), the AR+AR pair and the AR+diffusion pair with the lowest validation FAR are chosen, with FAR pooled using the step 9 weights. How often each pair wins across validation bootstrap resamples is reported (selection stability).
4. **Cost check:** measured GPU-seconds per candidate for both selected pairs, on the same GPU type and batch size (batch size 1 for both pairs if either includes iLLaDA). If they differ by more than 1.5×, the cheaper pair's members average their scores over T2 and, if needed, T3 (scored on validation and test before the preregistration freeze) until costs are within 1.5×, and this budget-matched comparison becomes the H1 test. If that cannot be built, H1 is reported without the "comparable cost" claim.
5. **Preregistration freeze:** the selected pairs, thresholds, answer-slot lengths, any budget-matching and the analysis code are committed with a hash before test labels are joined.

### On test (run once)

6. Apply the frozen thresholds and report each pair's **TAR and FAR with counts and 95% confidence intervals**, and **ΔFAR = FAR(AR+diffusion) − FAR(AR+AR)** (negative favors the mixed pair).
7. **Matching check:** because thresholds are frozen, test TAR will drift from 90%. The comparison counts as matched only if both pairs' test TAR are within ±3 points of 90% and within 3 points of each other (point estimates; intervals reported). Otherwise the matched comparison is reported as inconclusive and trade-off curves are shown instead. Nothing is retuned on test labels.
8. **Uncertainty:** paired bootstrap that resamples **whole problems** (keeping all candidates of a problem together, stratified by domain), 10,000 replicates, fixed seed, with thresholds held fixed. A secondary analysis repeats the validation-stage selection inside the bootstrap to show how much the choice of pairs itself varies.
9. **Pooling:** math and code are combined with each domain weighted by its number of wrong test answers, which is fixed by the frozen bank before unblinding, so the weights cannot move with the results. This gives the most precise pooled estimate. Each domain is also reported separately. Code enters the primary result only if it passes the code rule in §7.

### Decision rule (fixed in advance)

Smallest effect worth caring about: **3 points of FAR** at the matched operating point. Rows are checked in order; the first that applies is the verdict.

| 95% CI for ΔFAR (test) | Verdict |
|---|---|
| Entirely below 0, point estimate ≤ −3 points | AR+diffusion pair better, by a meaningful margin |
| Entirely below 0, point estimate > −3 points | Better, but by less than the meaningful margin |
| 90% CI inside (−3, +3) points | No meaningful benefit (equivalence) |
| Entirely above 0 | AR+diffusion pair worse |
| Otherwise | Inconclusive; report the interval and the smallest effect the data could have detected |

### Reference rows in the same table

- The best single verifier at its validation threshold for 90% TAR (does any pair beat one good verifier?).
- **Same-model pseudo-pairs:** one verifier with templates T1 and T2, fused the same way. A cross-model pair's gain counts as diversity only if it beats the pseudo-pair of its stronger member (by validation AUROC) at the same 90% pair TAR. It reuses scores already computed, so it costs nothing.
- **Ideal thresholds (test-tuned):** the same comparison with thresholds tuned on test, to show how much frozen thresholds cost. Secondary; no verdict uses it.
- Qwen3-8B paired with the AR and diffusion partners that give the lowest validation FAR (strong-baseline check).

### Secondary results (descriptive)

- **Accuracy-versus-cost curves** (PROJECT.md's ensemble gain curves): FAR at the matched operating point against measured GPU-seconds, for single verifiers and all pairs (no ensembles larger than two). The critique-then-verdict variants appear as AUROC against cost, since they have no validation thresholds.
- The 80% and 95% TAR operating points, and all eligible pairs, not just the selected two.
- **Breakdowns** by task, generator and error type, as in PROJECT.md. Small groups are descriptive only.

### H2 — mechanism

On wrong test answers, for every eligible pair plus the E1 and H2-trend pairs, at the pair's frozen 90%-pair-TAR thresholds (step 2):
- **κ on false accepts** (chance-adjusted agreement on which wrong answers get accepted), with bootstrap intervals;
- Spearman correlation of scores on wrong answers (no threshold needed);
- **CAPA** (Goel et al., 2025) from the verdict probabilities, implemented from the paper and unit-tested; secondary, because it depends on calibration, which differs between model classes.

**Competence check:** across cross-family AR–AR pairs (including the two small AR models, so competence varies), fit κ against the pair's mean AUROC. AR–diffusion pairs count as "more diverse" only if they sit below that trend, with an interval excluding zero. Low overlap with a weak verifier alone is not evidence. Reported by task and error type as well.

### E1 — shared-lineage control

Compare Qwen2.5-7B + Dream with Qwen2.5-7B + LLaDA/iLLaDA and with Qwen2.5-7B + each AR model, on FAR and κ. If Qwen+Dream is as decorrelated as Qwen+LLaDA, the difference survives shared initialization; if it looks like an AR pair, lineage or data likely matters. Not causal: Dream also has 580B extra training tokens and its own fine-tuning data. E1 runs if Dream passes the usable-scores gate; if Dream fails gate A, E1 is still reported, but descriptively only.

---

## 6. Sample size

- **Precision check after the pilot:** from pilot data, estimate how often the two candidate pairs disagree on wrong answers and how much answers to the same problem cluster. Then compute the smallest ΔFAR the planned test set can detect.
- **Planning figures (est.):** a 5,000-candidate bank gives about 1,650 wrong test answers. With the pooling in §5 step 9, the detectable effect is then roughly 3.2–4.6 points, depending on how often the pairs disagree. So even the top of PROJECT.md's range may not reach the 3-point margin, and the "no meaningful benefit" verdict (90% interval inside ±3 points) is reachable only if the observed difference is near zero.
- **Before the bank is frozen, the team picks one:** (a) stay within the approved range and report the smallest detectable effect alongside the verdict; or (b) ask the instructor, at the midterm, about a bank above 5,000 (§12). Equal domain weights would make this worse (about 3.8–5.4 points), which is why §5 weights by wrong answers.
- **Narrower results are less precise:** if the primary result is math-only, a 5,000-candidate bank gives roughly 3.6–5.3 points; a code-only result (a few hundred wrong test answers) can detect only large effects and is descriptive in practice (est.).

---

## 7. Pilot gates and decisions

Gate values are locked before pilot results are inspected. They are project-management thresholds, not scientific standards. The template is fixed (T1) and answer-slot lengths are set on the development bank, so the pilot gates are not judged on the data used to tune them.

| Gate | Pass | If it fails |
|---|---|---|
| **Data and pipeline** | 500 labeled pilot candidates; split and provenance checks pass; audit within target; scoring-code tests 1–3 pass for every model (under the §10 reduced pilot, the audit and the code half follow later). Pilot AUROC, calibration (ECE) and runtime are reported per verifier, as the proposal's Milestone 1 promised. | Fix the pipeline before judging any model |
| **Usable scores** | Median Yes+No probability ≥ 0.5 under T1 for every verifier, checked on the development bank before the pilot (and reported on the pilot) | Switch the answer words to "Correct/Incorrect" for every verifier and every template, once, on the development bank, so the wording stays identical across verifiers. A verifier that still fails is excluded. |
| **Gate A — competence** (PROJECT.md; required of diffusion models, and the same bar decides AR eligibility and the code rule) | Per domain, on T1 scores: AUROC ≥ 0.60 with 95% lower bound > 0.50; diffusion models must also be **within 0.10 AUROC of the small AR verifier** (Llama-3.2-3B). Re-checked on validation for every eligible diffusion model before selection (§5 step 3). | Not eligible in that domain. A model that fails the validation re-check is dropped from selection in that domain, and the drop is recorded in `decisions/`. If no diffusion model passes in any domain, use the one repair below. |
| **Gate B — prompt stability** (PROJECT.md) | AUROC spread across T1–T3 ≤ 0.05 for each verifier, per domain and pooled, on the pilot | Investigate the scoring. Any verifier, AR or diffusion, that stays unstable is excluded from primary pairs. Templates are never picked on pilot or test data. |
| **Dynamic range** | The best single AR verifier's pilot FAR at 90% TAR is between 0.10 and 0.70 per domain (target 0.15–0.60) | Rebalance toward harder or easier problems before freezing the bank |
| **No shortcuts** | A length-only score and a surface-feature score (answer format, number of reasoning steps or lines of code, hedging phrases; fit on the development bank) each reach AUROC ≤ 0.65 on the pilot. Problem- and generator-level scores, including problem difficulty, are uninformative by construction, because every problem and generator supplies equal numbers of correct and wrong answers. | Tighten the matched-pair filters and re-check |
| **Compute** | Measured settings fit the confirmed GPU hours with 30% contingency | Cut in the order of §9 |

**Code rule (single rule):** code enters the primary result only if, on the pilot, at least two of the four AR verifiers and at least one diffusion verifier pass the competence bar in code. Otherwise the primary result is math-only and code is reported descriptively.

**Decisions:**
- **Continue** if at least one diffusion model passes gates A and B in at least one domain. Passing does not require beating AR+AR in the pilot.
- **Repair once** (time-boxed to one week of one person) if no diffusion model passes: try the reconstruction score as that model's score. If it passes, the H1 test is declared in the preregistration as "AR verdict versus diffusion reconstruction score" and reported under that name — the interface change is never silent.
- **Pivot** if nothing passes after the repair: the **AR-family diversity study** on the same bank, harness and metrics — which AR families' mistakes differ most, and how close cross-family pairs come to a perfect verifier — plus the reported negative result "the tested diffusion models are not competent verdict verifiers on these tasks."
- If diffusion works on only one task, claims are restricted to that task.

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

Rows are read in order. An H2 result above the trend is read as "on the trend". If H2 is inconclusive, the H1 reading stands alone, with no mechanism claim.

E1 is read alongside: it can support or weaken a lineage explanation, but never proves one. If math and code disagree, claims are per domain.

---

## 9. Compute and reproducibility

- **Compute:** Colab Pro+ now; USC CARC allocation as the likely upgrade; modest API credits for generation. Every model fits one 24 GB GPU (measure in Phase 0).
- **Rough budget (est.):** candidate generation ≈ 10–20 A100-hours; primary scoring of ~10 models × ~5,000 candidates × 2 templates (plus 3 on the pilot) ≈ 5–15 A100-hours; reconstruction score on 3 diffusion models × ~2,000 candidates × ~32 Monte Carlo passes ≈ 5–15 A100-hours (about 2.5–3× more if all three diffusion models are repaired on the full bank); critique-then-verdict ≈ 10–20 more. Replace with Phase 0 measurements, and confirm Colab compute units and CARC status, since the total can exceed one Colab Pro+ subscription.
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
- **Release:** as in PROJECT.md, the labeled bank plus the overlap-analysis code is the project's public artifact, which survives any outcome. To keep it releasable, generator and dataset terms are checked in Phase 0 (for example GSM8K-Platinum's added annotations are CC BY-SA 4.0, and Falcon3 has its own license), and any generator whose outputs cannot be released is swapped before generation.

**Cut order if behind:** (1) critique-then-verdict ablation, (2) Qwen3 strong-baseline check, (3) answer-only control, (4) reconstruction ablation reduced to the pilot, 500 validation and 500 test candidates, (5) code domain. **Never cut:** the validation/test separation and freeze, the primary comparison, the audit, the same-model pseudo-pairs, gates A and B, and the generator and error-type breakdowns (a Milestone 2 commitment). Dropping the reconstruction ablation entirely, or the code domain, changes approved scope and is raised with the instructor first (§12).

---

## 10. Course deadlines and reporting

Fixed course dates, from PROJECT.md (check against Brightspace):

| Date | Deliverable |
|---|---|
| Oct 20–22 | Midterm presentation: 2 minutes, 2 slides — problem, method, how the instructor feedback was addressed, first pilot numbers or gate status |
| **Oct 27** | **Midterm report (4 pages):** how the instructor feedback was addressed (§5); pilot results and the gate outcome ("we defined a go/no-go gate; here is what passed it"); and the scope changes the instructor should know about (§12) |
| **Nov 20** | **Milestone 2:** full comparison complete, with robustness checks across generators and error types |
| Nov 20 – Dec 3 | Final presentations: 6 minutes, 5 slides |
| **Dec 15, 6:30 PM** | **Final report (8 pages + appendix).** No late tokens. |

- **Phase 0 (setup, before the pilot; owner F):** access granted for Llama-3.1, Llama-3.2 and Gemma-2; Falcon3 and GSM8K-Platinum terms checked; the four environments built; peak memory measured on the actual GPU; Colab compute units and CARC status confirmed. Every item this plan marks "Phase 0" belongs here.
- **Pilot status:** the self-imposed Oct 12 pilot milestone has passed and its status is not recorded in this folder. By Oct 21, F records in `decisions/` whether the pilot has run and whether anyone has seen pilot results. If results were seen before this plan (Oct 19), the midterm says the gate values were set afterwards.
- **Reduced pilot if needed:** if the pilot has not run, the Oct 27 pilot is cut to 250 math candidates, the four AR models plus Llama-3.2-3B, LLaDA and iLLaDA, templates T1–T3, and gates A and B. Scoring-code tests 1–3 still pass first for every included model. The data gate is judged on this subset without the audit, and gate outcomes are reported as provisional until the audit passes. The audit, the 250-candidate code pilot (which decides the code rule) and Dream follow under the same locked gate values, before the full bank is built. The midterm then reports the planned design and gate definitions (frozen only after validation, §5 step 5) plus the reduced pilot's results.
- **Minimum result for Nov 20:** H1 on math with the AR and diffusion models that pass the gates, with generator and error-type breakdowns (Milestone 2 promises them). If code is not in the Nov 20 result, the instructor is told (§9, §12), and the report says whether code will be added by Dec 15. Cut-order items (1)–(4) are optional.
- **What goes in the 8 pages:** question and design; bank and labels; verifiers and scoring; the primary H1 table and figure; H2 with the competence check; E1; limitations. **Appendix:** gates and pilot, full pair matrix, ablations, robustness, error types, prompts, reproducibility details.
- **Text reused from the proposal** gets the corrections recorded in the team's notes on the submitted version (the restored gate, grammar and punctuation).

---

## 11. Team roles (PROJECT.md's six roles)

Each owner keeps their code, a methods paragraph and one report-ready result; a reviewer from another role approves merges.

| Role | Owns | Reviewer |
|---|---|---|
| **A — Bank: math** | GSM8K/MATH sources, generators, checkers, matched-pair selection, math audit and error tags | E |
| **B — Bank: code** | HumanEval+/MBPP+, sandboxed execution, flaky-test reruns, code labels and error types | F |
| **C — Verifier harness** | Common scoring interface, prompt templates, AR adapters, Yes/No sets, batching and resume, cost logging | D |
| **D — Diffusion specialist** | LLaDA, iLLaDA and Dream adapters (mask ids, Dream's shift, answer slots), scoring-code tests, reconstruction ablation | C |
| **E — Metrics and statistics** | Validation selection and thresholds, preregistration freeze, bootstrap, decision rule, H2 and E1, power check, figures | C (A and B never review selection or thresholds, §2) |
| **F — Infrastructure and writing** | Environments, GPU access, run ledger, **test-label custody**, preregistration commit, report and slide integration, release | B |

Two short meetings a week: an integration demo and a results/blockers review. Design changes before the preregistration freeze are recorded in `decisions/`; after it, changes are allowed only as labeled post-hoc analyses.

---

## 12. Risks

| Risk | Control |
|---|---|
| Diffusion verifiers too weak or badly calibrated | Gates A and B; one repair; declared AR-only pivot |
| Silent scoring bugs (mask ids, Dream's shift, padding with bidirectional attention, library versions) | Scoring-code tests (§4); separate environments; cross-review between C and D |
| Confounding from data, post-training or lineage | Claims about tested pairs only; E1 shared-lineage control; generators outside verifier families |
| Label noise | Programmatic labels, stronger code tests, GSM8K-Platinum, audit |
| Too little data to detect 3 points | Pilot precision check; bank size decided before freezing (§6) |
| Memorized benchmark answers | Exposure differs by model and is unknown; the answer-only control is reported, and claims are scoped |
| Compute shortfall | Everything fits one 24 GB GPU; cut order (§9) |
| Changes the course did not approve | §13 lists them all. The midterm report names the ones that touch approved scope and asks the instructor to flag objections: four AR families instead of 2–3; three diffusion verifiers instead of 1–2, plus small AR references and a Qwen3 check; the validation/test split with a pilot kept separate from the full bank; a possible math-only primary result under the code rule; a bank above 5,000 if needed (§6); and no causal claims. |

---

## 13. Changes from PROJECT.md

Each change has one of four sources: **Feedback** (instructor), **Review** (a flaw confirmed by independent reviews), **Release** (a model released since), or **Course** (course logistics).

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
| 9 | AR: 2–3 families (Qwen2.5, Llama-3.1, Gemma-2 or Mistral) | All four of those families | Six AR+AR pairs instead of three makes the baseline harder to beat; one extra 7B pass per answer | Review |
| 10 | "Within striking distance of a small AR verifier" | Small AR verifiers named (Llama-3.2-3B, Qwen2.5-1.5B); gate A measured against Llama-3.2-3B | Makes gate A checkable; also gives H2 a competence range | Review |
| 11 | — | Qwen3-8B as a secondary strong-baseline check | Answers "the AR baseline is outdated" without breaking competence matching | Review |
| 12 | — | Same-model pseudo-pairs as a reference | Separates real diversity from prompt noise; free | Review |
| 13 | Negative result is informative | Fixed 3-point margin and a decision table including "no meaningful benefit" | Makes a negative result a testable outcome | Review |
| 14 | Record GPU-seconds | Measured cost must be within 1.5×, else the budget-matched comparison decides H1 | Makes "comparable inference cost" checkable | Review |
| 15 | Gate A (for diffusion verifiers) and gate B | Kept and made numeric. Gate A judged per domain on T1 and re-checked on validation before selection; its competence bar now also decides AR eligibility and the code rule. Gate B judged per domain and pooled, for every verifier. Added scoring-code tests, usable-scores, dynamic-range, no-shortcut and compute gates; answer-slot lengths set on a separate development bank | Gates that cannot pass by chance (gate A's lower bound), hide a domain failure, or be judged on the data used to tune them | Review |
| 16 | Pivot to AR-family study | Kept; one declared repair (reconstruction score) first, never silent | The proposal's motivation for reconstruction scoring | Review |
| 17 | Shared-lineage control where feasible; any AR+diffusion pair | Qwen2.5-7B-Instruct + Dream as E1 (Dream is initialized from the Qwen2.5-7B base); Dream × Qwen2.5 is excluded from H1 | Available at no extra cost; a shared-lineage pair would blur the cross-boundary comparison | Review |
| 18 | CAPA / κ on mistakes | κ on false accepts and score correlation first, on wrong answers only (false-accept overlap); CAPA secondary; competence trend check | Joint false accepts are what H1 measures; CAPA depends on calibration, which differs between model classes | Review |
| 19 | 2,000–5,000 candidates | Same range; target set by the pilot precision check; growing beyond it would be raised with the instructor | The 3-point margin may need the top of the range | Review |
| 20 | Dated timeline; pilot covers both task types and all models | Course deadlines kept (§10); internal dates set by the team; a reduced pilot (math only; code and Dream follow) if the full one cannot finish by Oct 27 | Oct 12 pilot milestone has passed | Course |
| 21 | 500-item pilot grows into the 2,000–5,000 bank | Pilot and development bank kept separate from the full bank | The data used to set gates and tune lengths stays out of the final test | Review |
| 22 | Bank never modified mid-experiment | Corrections before the freeze create new versions; after unblinding, the primary result stays on the preregistered version | Same aim, made workable when a checker bug is found early | Review |
| 23 | Math and code both in the main comparison | Code enters the primary result only under the §7 code rule; the primary result is narrowed to one domain if no diffusion model is eligible in every covered domain; dropping code entirely is raised with the instructor | Avoids pooling a domain where the verifiers are not competent | Review |
| 24 | Weighting of math and code unspecified | Domains weighted by their number of wrong test answers, fixed before unblinding | Most precise pooled estimate; equal weights would need a larger bank | Review |
| 25 | If compute tightens, shrink the bank or model count | Cut order in §9, which trims ablations first and never drops the generator and error-type breakdowns | Protects the primary comparison and Milestone 2 commitments | Review |
| 26 | Causal claims allowed if the controls support them | No causal claims; E1 can support or weaken a lineage explanation only | The available control (Dream vs Qwen2.5) also differs in training data and fine-tuning, so it cannot isolate the objective | Review |
| 27 | API credits for any closed-model AR baselines (optional) | Not used; API credits go to one generator | Budget, and closed models cannot be scored reproducibly | Course |
| 28 | — | Added: answer-only control, critique-then-verdict ablation, label audit, development bank, generation defaults, AR arm of the reconstruction score, input-length and per-problem caps, the generator-share cap and quality/consistency filters, MATH limited to checkable answers, usable-scores and compute gates, label custody, a fallback if the AR pool shrinks, extra repository folders; accuracy-versus-cost curves cover single verifiers and pairs only | Each closes a gap a reviewer identified; none changes the question | Review |

**Considered and not adopted** (from an independently written alternative plan): newer AR models as the primary pool (breaks competence matching), DiffusionGemma and Gemma 4 26B (needs an 80 GB GPU), Nemotron-Labs-Diffusion, a five-model competence ladder, an 8,000-candidate bank, and an AceInstruct-based lineage contrast. They add scope the course format cannot hold. The alternative plans and their reviews are kept with the team's working files, outside this repository.

---

## 14. References (verify exact citations before each report)

- Saad-Falcon et al., 2025 — *Shrinking the Generation-Verification Gap with Weak Verifiers* (Weaver). Note: its "correlated biases on hard examples" is the authors' explanation for diminishing gains, not a measurement.
- Goel et al., ICML 2025 — *Great Models Think Alike and this Undermines AI Oversight* (CAPA).
- Nie et al., 2025 — *Large Language Diffusion Models* (LLaDA); iLLaDA model card and paper (2026); Ye et al., 2025 — Dream 7B.
- Panickssery et al., 2024 — *LLM Evaluators Recognize and Favor Their Own Generations* (generator choice).
- PRISM, DEER, HIVE, FreeCache — diffusion self-checking and drafting directions, summarized in the team's vetting notes. Their exact arXiv IDs are not recorded anywhere yet: look them up and add them here before the midterm report.
- Task sources: GSM8K (Cobbe et al., 2021) and GSM8K-Platinum (Vendrow et al., 2025); MATH (Hendrycks et al., 2021); MBPP (Austin et al., 2021) and HumanEval (Chen et al., 2021), with EvalPlus (Liu et al., 2023).
