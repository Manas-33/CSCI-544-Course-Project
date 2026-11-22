"""Cells of notebook 13 (scoring the full bank). make_notebooks.py writes them to 13_score_full.ipynb.

The guards, the scoring commands, saving and the completeness check are in harness/full_run.py.
"""

TITLE = '''
# Full bank · score both banks with every verifier (T1 and T2)

Runtime: **A100 or L4, the same type every time.** The first saved scores fix it for every verifier and both banks (`gpu.txt`; `decisions/2026-11-22-scoring-one-gpu-type.md`). If the repair was chosen, notebook 12 scores the whole math bank on the same type: about 4.5 hours on an A100, about 14 on an L4.

**Always the same Google account.** The GPU type and the finished runs are remembered in that account's Drive, so all full-bank scoring (this notebook, and notebook 12 on full-bank answers) runs from one account.

Before *Runtime → Run all*:
1. `main` has the **final model record** and `configs/final_models.json` (PLAN.md §7).
2. `main` has the **frozen bank** you want to score: `bank/full/<math or code>/bank.jsonl` and `manifest.json` (PLAN.md §2). Math and code can be scored one at a time: set `DOMAINS`.
3. The Colab secret `HF_TOKEN` has access to Llama, Gemma and Mistral.
4. Choose the runtime type. Nothing is scored on another type than before; the message says what to do.

What it runs: 8 models, T1 and T2, batch size 1, BF16, no quantization, SDPA (Gemma-2 eager), 1 answer slot, with the pilot's model commits (`configs/full_*.json`, `harness/diffusion/scorer.py`). No label is read. About 3,000 answers take roughly 1¼ hours on an A100 and 3 on an L4 (estimated from the pilot), model downloads included.

**Every finished run is saved to Google Drive** (`MyDrive/cs544_full_scoring/full/`). After a disconnect, run all cells again: finished runs are skipped, an unfinished one is scored again, and the code stays on the commit of the first saved scores. The last cell disconnects the runtime.

**Until both banks are scored**, leave the scoring code on `main` as it is (`SCORING_CODE` in `harness/full_run.py`, with `requirements.txt` and `configs/full_*.json`): the second bank is refused if it changed. **To start over** (a corrected bank, or another GPU type), delete `MyDrive/cs544_full_scoring/` first, and only before the preregistration commit.'''

PARAMETERS = '''
DOMAINS = ["math", "code"]      # or just one of them
GROUPS = ["llama_gemma", "qwen_mistral", "small", "llada", "dream"]     # remove a group to skip it in this session
DISCONNECT = True               # release the runtime at the end
BANKS = "bank/full"             # a corrected bank is a new folder
DOMAINS_ARG = " ".join(DOMAINS)
STOPPED = "stopped: read the message above (REFUSED says why and what to do), then run this cell again"'''

SETUP = '''
from pathlib import Path
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive, userdata
from huggingface_hub import login
drive.mount("/content/drive")
login(token=userdata.get("HF_TOKEN"))
DRIVE = "/content/drive/MyDrive/cs544_full_scoring"
SAVED = f"{DRIVE}/full"
ALSO_GPU = "/content/drive/MyDrive/cs544/reconstruction/math/gpu.txt"   # notebook 12's full-bank scores, if any
!mkdir -p {SAVED}
![ -d /content/project ] || git clone -q REPO_URL /content/project
%cd /content/project
!git fetch -q origin
COMMITS = {p.read_text().strip() for p in (Path(SAVED) / d / "commit.txt" for d in DOMAINS) if p.exists()}
assert len(COMMITS) <= 1, "math and code were started on different commits: put one of them in DOMAINS"
COMMIT = COMMITS.pop() if COMMITS else "origin/main"     # a resumed run keeps the code of its first saved scores
!git checkout -q {COMMIT}
!git log --oneline -1'''

GUARDS = '''
!python -m harness.full_run start --domains {DOMAINS_ARG} --banks {BANKS} --saved {SAVED} --also-gpu {ALSO_GPU}
assert _exit_code == 0, STOPPED
!pip install -q -r requirements.txt'''

SCORE = '''
if "@GROUP@" in GROUPS:
    !python -m harness.full_run score --group @GROUP@ --domains {DOMAINS_ARG} --banks {BANKS} --saved {SAVED} --also-gpu {ALSO_GPU}
    assert _exit_code == 0, STOPPED
    !rm -rf /root/.cache/huggingface/hub/models--*'''

DIFFUSION = '''
if "@GROUP@" in GROUPS:
    !pip install -q uv
    !uv venv -q --clear --python 3.11 /content/env_@GROUP@
    !uv pip install -q --python /content/env_@GROUP@/bin/python @TORCH@ -r harness/diffusion/requirements/@GROUP@.txt
    PY = "/content/env_@GROUP@/bin/python"
    !{PY} -c "import sys, torch, transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__, torch.cuda.is_available())"
    !python -m harness.full_run score --group @GROUP@ --python {PY} --domains {DOMAINS_ARG} --banks {BANKS} --saved {SAVED} --also-gpu {ALSO_GPU}
    assert _exit_code == 0, STOPPED
    !rm -rf /root/.cache/huggingface/hub/models--*'''

CHECK = '''
!python -m harness.full_run check --domains {DOMAINS_ARG} --banks {BANKS} --saved {SAVED}
print("every score is there" if _exit_code == 0 else "NOT COMPLETE: run the missing groups or banks (this is expected after a partial run)")'''

FINISH = '''
!zip -qr /content/full_scores.zip results/full
!cp /content/full_scores.zip {DRIVE}/full_scores.zip
if DISCONNECT:
    drive.flush_and_unmount()
    from google.colab import runtime
    runtime.unassign()'''


def cells(repo):
    return [
        ("markdown", TITLE),
        ("code", PARAMETERS),
        ("code", SETUP.replace("REPO_URL", repo)),
        ("markdown", "## 1. Guards\nStops here unless the final model record is merged, the banks are frozen, the GPU type is the one of the first scores, and every saved score file was made with these settings. Prints what is done and what is left."),
        ("code", GUARDS),
        ("markdown", "## 2. Llama-3.1-8B and Gemma-2-9B (eager attention)"),
        ("code", SCORE.replace("@GROUP@", "llama_gemma")),
        ("markdown", "## 3. Qwen2.5-7B and Mistral-7B"),
        ("code", SCORE.replace("@GROUP@", "qwen_mistral")),
        ("markdown", "## 4. Llama-3.2-3B and Qwen2.5-1.5B"),
        ("code", SCORE.replace("@GROUP@", "small")),
        ("markdown", "## 5. LLaDA-8B, in its own Python 3.11 environment\nColab's Python cannot install LLaDA's pins (as in notebooks 02, 07 and 09). Expect 3.11, 2.6.0 and 4.38.2."),
        ("code", DIFFUSION.replace("@GROUP@", "llada").replace("@TORCH@", "torch==2.6.0")),
        ("markdown", "## 6. Dream-7B, in its own Python 3.11 environment\nExpect 3.11, 2.5.1 and 4.46.2."),
        ("code", DIFFUSION.replace("@GROUP@", "dream").replace("@TORCH@", "torch==2.5.1")),
        ("markdown", "## 7. Is everything there?\nEvery verifier × template × candidate, no failed or too-long row, seconds and memory recorded, one GPU type, and the final analysis' loader accepts the files. The report is saved beside the scores (`completeness_*.json`)."),
        ("code", CHECK),
        ("markdown", "## 8. Save everything and disconnect\nThe zip is `MyDrive/cs544_full_scoring/full_scores.zip`; unzip it into the repo (`results/full/`) and commit it when the check says complete. The last line gives the GPU back."),
        ("code", FINISH),
    ]
