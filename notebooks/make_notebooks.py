"""Generate the Colab notebooks in this folder. Run from the repository root:

    python notebooks/make_notebooks.py notebooks

Shell lines use {VAR} for Python variables: IPython reads {N}.json as attribute access and
then passes the whole line through unexpanded.
"""
import json
import sys
from pathlib import Path

REPO = "https://github.com/Manas-33/CSCI-544-Course-Project.git"

def nb(cells):
    out = []
    for kind, text in cells:
        src = text.strip("\n").splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        out.append(cell)
    return {"cells": out, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}

def setup(title, env_line, extra=""):
    return [
        ("markdown", title),
        ("code", f'''
BRANCH = "main"   # the branch to test; main once the stage-3 PR is merged
!nvidia-smi --query-gpu=name,memory.total --format=csv
!git clone -q -b {{BRANCH}} {REPO} /content/project
%cd /content/project
!git log --oneline -1
{env_line}
{extra}'''),
    ]

def finish(name):
    return [("markdown", f'''
## Download the results
This zips `results/stage3/{name}/` (scores, manifests and reports). Send the zip back, or unzip it into the repo at the same path.'''),
            ("code", f'''
!cd /content/project && zip -qr /content/stage3_{name}.zip results/stage3/{name}
from google.colab import files
files.download("/content/stage3_{name}.zip")''')]

DEV_HASH = '''import hashlib
DEV_HASH = hashlib.sha256(open("bank/dev/dev_bank.jsonl", "rb").read()).hexdigest()
SANITY_HASH = hashlib.sha256(open("bank/sanity/sanity_set.jsonl", "rb").read()).hexdigest()
print(DEV_HASH, SANITY_HASH)'''

ar = nb(setup('''
# Stage 3 · AR verifiers (Qwen, Mistral, Llama, Gemma, small models)

Runtime: **A100** (40 GB; the FP32 checks need it). One fresh runtime for this notebook only.
Llama, Gemma and Mistral are gated: accept their licenses on Hugging Face first, and add your token as the Colab secret `HF_TOKEN`.

What it runs (PLAN.md §4 and §7):
1. The AR reference check for Qwen and Mistral: the rendered prompts, our one-pass readout against the first step of the model's own generate(), and the same items in FP32 against BF16 (the check that would have caught the eager-attention bug). This is the evidence for signing off Mistral's sanity failure.
2. Qwen and Mistral on the 71 development pairs: a pipeline check (real tokenizer lengths, usable-scores gate), not a result.
3. Sanity check plus reference check for Llama-3.1-8B and Gemma-2-9B (Gemma with eager attention: SDPA skips its soft-capping).
4. Sanity check plus reference check for Llama-3.2-3B (gate A's small AR verifier) and Qwen2.5-1.5B.''',
'!pip install -q -r requirements.txt -r requirements-analysis.txt',
'# Add a Hugging Face token as a Colab secret named HF_TOKEN (key icon on the left) with access\n# to Llama, Gemma and Mistral; this cell then logs in before anything downloads.\nfrom google.colab import userdata\nfrom huggingface_hub import login\nlogin(token=userdata.get("HF_TOKEN"))\nOUT = "results/stage3/ar"') + [
 ("markdown", "## 1. Reference check: Qwen and Mistral"),
 ("code", '!python -m harness.ar_reference --config configs/ar_pair_sanity.json --input bank/sanity/sanity_set.jsonl --output {OUT}/reference_qwen_mistral.json --fp32-check'),
 ("markdown", "## 2. Qwen and Mistral on the development pairs (pipeline check)"),
 ("code", '''!python -m harness.run --config configs/ar_pair_dev.json --input bank/dev/dev_bank.jsonl --output-dir {OUT}/dev-qwen-mistral --device cuda:0 --dtype bf16 --no-load-in-4bit
!python -m harness.evaluate --bank bank/dev/dev_bank.jsonl --labels bank/dev/dev_labels.jsonl --run-dir {OUT}/dev-qwen-mistral --output-dir {OUT}/dev-qwen-mistral-analysis'''),
 ("markdown", "## 3. Llama-3.1-8B and Gemma-2-9B: sanity check and reference check"),
 ("code", '''!python -m harness.run --config configs/ar_sanity_llama_gemma.json --input bank/sanity/sanity_set.jsonl --output-dir {OUT}/sanity-llama-gemma --device cuda:0 --dtype bf16 --no-load-in-4bit
!python -m harness.evaluate --sanity --bank bank/sanity/sanity_set.jsonl --labels bank/sanity/sanity_set.jsonl --run-dir {OUT}/sanity-llama-gemma --output-dir {OUT}/sanity-llama-gemma-analysis
!python -m harness.ar_reference --config configs/ar_sanity_llama_gemma.json --input bank/sanity/sanity_set.jsonl --output {OUT}/reference_llama_gemma.json'''),
 ("markdown", "## 4. Llama-3.2-3B and Qwen2.5-1.5B: sanity check and reference check"),
 ("code", '''!python -m harness.run --config configs/ar_sanity_small.json --input bank/sanity/sanity_set.jsonl --output-dir {OUT}/sanity-small --device cuda:0 --dtype bf16 --no-load-in-4bit
!python -m harness.evaluate --sanity --bank bank/sanity/sanity_set.jsonl --labels bank/sanity/sanity_set.jsonl --run-dir {OUT}/sanity-small --output-dir {OUT}/sanity-small-analysis
!python -m harness.ar_reference --config configs/ar_sanity_small.json --input bank/sanity/sanity_set.jsonl --output {OUT}/reference_small.json --fp32-check'''),
 ("markdown", "A sanity evaluation that fails the 0.90 bar exits with code 1 and prints an error line; that is the result, not a crash. Keep going."),
] + finish("ar"))

def diffusion(model, title, env, llada_repo, restart_note="", name=None, prefixes=("none",), python311=False):
    """One diffusion notebook. With several answer prefixes (iLLaDA), the prefix is chosen on the
    development bank together with the slot count (decisions/2026-10-25-illada-thinking-off.md)."""
    name = name or model
    repo_flag = " --llada-repo /content/LLaDA" if llada_repo else ""
    clone = "\n!git clone -q https://github.com/ML-GSAI/LLaDA /content/LLaDA && git -C /content/LLaDA checkout -q 9182493" if llada_repo else ""
    if python311:
        # Colab's Python (3.13 as of Oct 2026) cannot install this environment's pins, so the
        # notebook builds its own Python 3.11 environment and runs every command with it.
        install = ("!pip install -q uv\n!uv venv -q --python 3.11 /content/env\n"
                   f"!uv pip install -q --python /content/env/bin/python torch==2.6.0 -r harness/diffusion/requirements/{env}.txt\n"
                   'PY = "/content/env/bin/python"\n'
                   '!{PY} -c "import sys, torch, transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__, torch.cuda.is_available())"')
    else:
        install = f"!pip install -q -r harness/diffusion/requirements/{env}.txt"
    cells = setup(title, install + clone, f'OUT = "results/stage3/{name}"\n!mkdir -p {{OUT}}\n' + DEV_HASH)
    if restart_note:
        cells.insert(2, ("markdown", restart_note))
    if model == "llada":
        cells += [
            ("markdown", "## 1. LLaDA's answer-slot count (1, chosen earlier on the development bank) and no answer prefix"),
            ("code", 'N = 1\nPREFIX = "none"'),
        ]
    elif len(prefixes) == 1:
        cells += [
            ("markdown", "## 1. Choose the answer-slot count on the development bank (PLAN.md §4)\nScores the 142 development answers with 1, 8 and 32 slots (labels are never read), then picks the highest median Yes+No mass, ties to fewer slots. Also reports the usable-scores gate."),
            ("code", f'''for n in (1, 8, 32):
    !python -m harness.diffusion.run_score --model {model} --n-slots {{n}} --items bank/dev/dev_bank.jsonl --bank-hash {{DEV_HASH}} --out {{OUT}}/dev_T1_n{{n}}.jsonl
!python -m harness.diffusion.choose_slots {{OUT}}/dev_T1_n1.jsonl {{OUT}}/dev_T1_n8.jsonl {{OUT}}/dev_T1_n32.jsonl --report {{OUT}}/slots.json
import json
choice = json.load(open(f"{{OUT}}/slots.json"))
N, PREFIX = choice["chosen_n_slots"], choice["chosen_answer_prefix"]
print("answer slots:", N, "| answer prefix:", PREFIX)'''),
        ]
    else:
        names = ", ".join(f'"{p}"' for p in prefixes)
        cells += [
            ("markdown", "## 1. Choose the answer prefix and slot count on the development bank\nScores the 142 development answers with every answer prefix and 1, 8 and 32 slots (labels are never read), then picks the combination with the highest median Yes+No mass; ties go to fewer slots, then to the simpler prefix. Also reports the usable-scores gate. The rule is fixed in `decisions/2026-10-25-illada-thinking-off.md`."),
            ("code", f'''for p in ({names}):
    for n in (1, 8, 32):
        !python -m harness.diffusion.run_score --model {model} --answer-prefix {{p}} --n-slots {{n}} --items bank/dev/dev_bank.jsonl --bank-hash {{DEV_HASH}} --out {{OUT}}/dev_T1_{{p}}_n{{n}}.jsonl
FILES = " ".join(f"{{OUT}}/dev_T1_{{p}}_n{{n}}.jsonl" for p in ({names}) for n in (1, 8, 32))
!python -m harness.diffusion.choose_slots {{FILES}} --report {{OUT}}/slots.json
import json
choice = json.load(open(f"{{OUT}}/slots.json"))
N, PREFIX = choice["chosen_n_slots"], choice["chosen_answer_prefix"]
print("answer slots:", N, "| answer prefix:", PREFIX)'''),
        ]
    cells += [
        ("markdown", "## 2. Smoke test on a tiny random-weight copy of the official code (CPU)"),
        ("code", f"!python -m harness.diffusion.checks.check_tiny --model {model}{repo_flag} --answer-prefix {{PREFIX}} --report {{OUT}}/tiny.json"),
        ("markdown", "## 3. Scoring-code tests 1–3 (PLAN.md §4)\nTest 2 compares our scorer with the official model code's first forward pass; test 1 is the sanity bar (AUROC ≥ 0.90); test 3 checks batched against unbatched scores and repeat runs."),
        ("code", f'''!python -m harness.diffusion.checks.check_parity --model {model} --n-slots {{N}} --answer-prefix {{PREFIX}}{repo_flag} --report {{OUT}}/test2_parity_n{{N}}.json
!python -m harness.diffusion.checks.check_sanity --model {model} --n-slots {{N}} --answer-prefix {{PREFIX}} --report {{OUT}}/test1_sanity_n{{N}}.json --out {{OUT}}/sanity_scores_n{{N}}.jsonl
!python -m harness.diffusion.checks.check_invariance --model {model} --n-slots {{N}} --answer-prefix {{PREFIX}} --report {{OUT}}/test3_invariance_n{{N}}.json --save {{OUT}}/invariance_scores_n{{N}}.json'''),
    ]
    if model == "llada":
        cells += [
            ("markdown", "## 4. Agreement with the saved LLaDA run\nScores the 50 sanity items unbatched with this scorer and compares them with the run in `results/llada-comparison-20261019-213107` (every score within 0.05, Spearman above 0.999)."),
            ("code", '''!python -m harness.diffusion.run_score --model llada --n-slots 1 --batch-size 1 --items bank/sanity/sanity_set.jsonl --bank-hash {SANITY_HASH} --out {OUT}/sanity_T1_n1.jsonl
!python -m harness.llada_compare --parity {OUT}/sanity_T1_n1.jsonl | tee {OUT}/agreement_with_saved_run.json'''),
        ]
    cells.append(("markdown", "Each check prints PASS or FAIL and writes its report even when it fails; a FAIL is a result, not a crash. Keep going."))
    if python311:
        cells = [(kind, text.replace("!python -m ", "!{PY} -m ") if kind == "code" else text) for kind, text in cells]
    return nb(cells + finish(name))

llada = diffusion("llada", '''
# Stage 3 · LLaDA-8B-Instruct: scoring-code tests

Runtime: **A100** (the saved LLaDA run used an A100 with torch 2.6.0; matching it keeps BF16 scores comparable). One fresh runtime for this notebook only.

LLaDA already has a saved sanity run (AUROC 0.885, fails the 0.90 bar). This notebook runs what is still missing: the parity test against the official code (test 2), batch invariance (test 3), the sanity check through the shared scorer, and agreement with the saved run.

**Its own Python.** Colab's Python 3.13 cannot install LLaDA's pinned packages (`sentencepiece 0.2.0` and `numpy<2` have no Python 3.13 builds). The first code cell therefore builds a separate Python 3.11 environment with `uv`, using torch 2.6.0 as the saved run did, and every command runs with it (`{PY}`). It prints its Python, torch and transformers versions: expect 3.11, 2.6.0 and 4.38.2.''', "llada", True, python311=True)

illada = diffusion("illada", '''
# Stage 3 · iLLaDA-8B-Instruct with thinking off: answer prefix, slots and scoring-code tests

Runtime: **L4 or A100**. One fresh runtime for this notebook only. iLLaDA runs on LLaDA's official code (`mask_id=5`) and is scored unbatched, so test 3 checks repeat runs only.

The first run (`results/stage3/illada/`) failed the usable-scores gate: iLLaDA opens every answer with a `<think>` block, so its first answer slot holds `<`, not Yes/No. As recorded in `decisions/2026-10-25-illada-thinking-off.md`, this run turns thinking off with an empty think block after the generation prompt, choosing the exact text and the slot count on the development bank by median Yes+No mass. Results go to `results/stage3/illada-thinking-off/`.''', "illada", True,
                  name="illada-thinking-off", prefixes=("none", "think_nl", "think_nlnl"))

dream = diffusion("dream", '''
# Stage 3 · Dream-v0-Instruct-7B: answer slots and scoring-code tests

Runtime: **A100** (Dream needs at least 20 GB). One fresh runtime for this notebook only. Dream's logits are shifted right by one position; test 2 is also the shift check.''', "dream", False,
restart_note='''**After the install cell finishes:** Dream pins torch 2.5.1, which replaces Colab's torch. Open *Runtime → Restart session*, then run the next cell, which goes back into the project folder.''')
# after restart, cwd and variables are lost: add a re-entry cell right after the note
for i, c in enumerate(dream["cells"]):
    if c["cell_type"] == "markdown" and "Restart session" in "".join(c["source"]):
        reentry = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                   "source": ['%cd /content/project\n', 'OUT = "results/stage3/dream"\n', '!mkdir -p {OUT}\n',
                              *[l + "\n" for l in DEV_HASH.splitlines()],
                              'import torch, transformers\n', 'print(torch.__version__, transformers.__version__)   # expect 2.5.1 and 4.46.2']}
        dream["cells"].insert(i + 1, reentry)
        break

out = Path(sys.argv[1])
for name, book in (("01_ar_verifiers", ar), ("02_llada", llada), ("03_illada", illada), ("04_dream", dream)):
    (out / f"{name}.ipynb").write_text(json.dumps(book, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", name)
