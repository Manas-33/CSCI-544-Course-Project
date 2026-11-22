"""Cells of notebook 12 (the backup score). make_notebooks.py writes them to 12_backup_score.ipynb."""

TITLE = '''
# Backup score · reconstruction score (LLaDA, Dream) and the AR arm

Runtime: **A100 or L4** (an L4 is roughly three times slower per pass). One fresh runtime. Colab secret `HF_TOKEN` (access to Llama, Gemma and Mistral) is needed for the AR arm only.

**Sections 1 and 2 are the only part the repair needs** (PLAN.md §7): LLaDA's reconstruction score on the pilot's 250 math answers, 128 Monte Carlo samples, seeds 544 and 545, then the pass/fail report. About 70 minutes on an A100 at one sample per forward pass (estimated from the pilot's 0.063 s per pass), so roughly 3.5 hours on an L4. Everything else is off until you set its switch in the first code cell.

Results go to Google Drive (`MyDrive/cs544/reconstruction`) as each answer is scored. If the runtime stops, run the notebook again: finished answers are skipped. **Stay on one GPU type:** both seeds must come from the same one, so the first run's type is kept in `gpu.txt` there, and on another type the pilot and development sections are skipped. Full-bank answers are scored on one GPU type too, A100 or L4: the type of notebook 13's scores and of the first answers scored here (`math/gpu.txt`; `decisions/2026-11-22-scoring-one-gpu-type.md`), and on another type they are skipped. The last cell disconnects the runtime. No test label is read here; the only labels read are the pilot's, in the report.'''

PARAMETERS = '''
BRANCH = "main"
RUN_PILOT_CODE = False   # LLaDA on the pilot's code answers (descriptive), about 50 minutes (A100)
RUN_DEV = False          # LLaDA on the 142 development answers, two seeds, for the sample count; about 30 minutes (A100)
RUN_DREAM = False        # Dream: the same sections (information only)
RUN_AR = False           # exact log-probability with the four AR models (information only), a few minutes each
# Full-bank answers are scored only after the final model record is merged (PLAN.md section 7), on one GPU type.
BANK = ""                # "bank/full/math/bank.jsonl" to score full-bank answers; "" = none
WHOLE_BANK = False       # False: 500 validation + 500 test answers of the math bank (the side experiment)
                         # True: every answer of the math bank, only if LLaDA's score is used as the repair
N = 128                  # Monte Carlo samples for the side experiment (the development-bank choice may be smaller)
DREAM_SEEDS = (544, 545)'''

SETUP = '''
import glob, os, subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
GPU = "A100" if "A100" in gpu else "L4" if "L4" in gpu.split() else None
assert GPU, f"{gpu or 'no GPU'}: choose an A100 or an L4 (Runtime > Change runtime type)"
from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/cs544/reconstruction"
!mkdir -p {OUT}/pilot {OUT}/dev
if not glob.glob(f"{OUT}/pilot/*.jsonl") + glob.glob(f"{OUT}/dev/*.jsonl"):   # nothing scored yet
    with open(f"{OUT}/gpu.txt", "w") as f:
        f.write(GPU)
with open(f"{OUT}/gpu.txt") as f:
    PILOT_GPU = f.read().strip()
SAME_GPU = GPU == PILOT_GPU   # the pilot and development scores all come from one GPU type
if not SAME_GPU:
    print(f"The scores in {OUT} were made on {PILOT_GPU}; this is {GPU}. The pilot and development sections are skipped; delete the folder to start over on {GPU}.")
![ -d /content/project ] || git clone -q -b {BRANCH} REPO_URL /content/project
%cd /content/project
!(date -u; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; git rev-parse HEAD) | tee -a {OUT}/runs.txt'''

LLADA_ENV = '''
!pip install -q uv
![ -d /content/env_llada ] || uv venv -q --python 3.11 /content/env_llada
!uv pip install -q --python /content/env_llada/bin/python torch==2.6.0 -r harness/diffusion/requirements/llada.txt
![ -d /content/LLaDA ] || git clone -q https://github.com/ML-GSAI/LLaDA /content/LLaDA
!git -C /content/LLaDA checkout -q 9182493
PY = "/content/env_llada/bin/python"
!{PY} -c "import sys, torch, transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__, torch.cuda.is_available())"'''

LLADA_CHECK = '''
import json
!rm -f {OUT}/check_llada.json
!{PY} -m harness.diffusion.checks.check_reconstruction --model llada --llada-repo /content/LLaDA --report {OUT}/check_llada.json
OK = os.path.exists(f"{OUT}/check_llada.json") and json.load(open(f"{OUT}/check_llada.json"))["status"] == "PASS"
print("LLaDA matches the official code:", OK)'''

LLADA_PILOT = '''
if OK and SAME_GPU:
    for s in (544, 545):
        !{PY} -m harness.diffusion.run_reconstruction --model llada --domain math --seed {s} --items bank/pilot/pilot_bank.jsonl --out {OUT}/pilot/llada_math_s{s}.jsonl'''

REPORT = '''
!python -m analysis.reconstruction pilot --bank bank/pilot/pilot_bank.jsonl --labels bank/pilot/pilot_labels.jsonl --scores {OUT}/pilot --verdicts results/pilot/ar results/pilot/llada results/pilot/dream --out {OUT}/pilot/analysis'''

LLADA_OPTIONAL = '''
if OK and SAME_GPU and RUN_PILOT_CODE:
    for s in (544, 545):
        !{PY} -m harness.diffusion.run_reconstruction --model llada --domain code --seed {s} --items bank/pilot/pilot_bank.jsonl --out {OUT}/pilot/llada_code_s{s}.jsonl
if OK and SAME_GPU and RUN_DEV:
    for s in (544, 545):
        !{PY} -m harness.diffusion.run_reconstruction --model llada --seed {s} --items bank/dev/dev_bank.jsonl --out {OUT}/dev/llada_s{s}.jsonl
    !python -m analysis.reconstruction samples --bank bank/dev/dev_bank.jsonl --scores {OUT}/dev --out {OUT}/dev/samples.json'''

BANK_SETUP = '''
NAME = os.path.basename(os.path.dirname(BANK)) if BANK else ""
if BANK and NAME != "math":
    print("full-bank answers are scored for math only (PLAN.md section 4): nothing scored")
    BANK = ""
if BANK:   # one GPU type for all full-bank scores: notebook 13's (in this Drive, or committed), and those already here
    KEPT = [f"{OUT}/{NAME}/gpu.txt"] if glob.glob(f"{OUT}/{NAME}/*/*.jsonl") else []
    KEPT += glob.glob("/content/drive/MyDrive/cs544_full_scoring/full/gpu.txt") + glob.glob("results/full/gpu.txt")
    TYPES = set()
    for path in KEPT:
        with open(path) as f:
            TYPES.add(f.read().strip())
    if TYPES - {GPU}:
        print(f"full-bank answers were scored on {' and '.join(sorted(TYPES))}; this is {GPU}: nothing scored. Choose that runtime type.")
        BANK = ""
    else:
        !mkdir -p {OUT}/{NAME}
        with open(f"{OUT}/{NAME}/gpu.txt", "w") as f:
            f.write(GPU)
SUBSET = f"{OUT}/{NAME}/side/subset.json"
if BANK and not WHOLE_BANK:
    !mkdir -p {OUT}/{NAME}/side
    !python -m analysis.reconstruction draw --bank {BANK} --out {SUBSET}'''

LLADA_BANK = '''
if OK and BANK and WHOLE_BANK:
    !{PY} -m harness.diffusion.run_reconstruction --model llada --seed 544 --n-samples 128 --items {BANK} --out {OUT}/{NAME}/whole/llada_n128_s544.jsonl
if OK and BANK and not WHOLE_BANK:
    !{PY} -m harness.diffusion.run_reconstruction --model llada --seed 544 --n-samples {N} --items {BANK} --subset {SUBSET} --out {OUT}/{NAME}/side/llada_n{N}_s544.jsonl'''

DREAM_ENV = '''
DREAM_OK = False
if RUN_DREAM:
    ![ -d /content/env_dream ] || uv venv -q --python 3.11 /content/env_dream
    !uv pip install -q --python /content/env_dream/bin/python -r harness/diffusion/requirements/dream.txt
    PY = "/content/env_dream/bin/python"
    !{PY} -c "import sys, torch, transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__, torch.cuda.is_available())"
    !rm -f {OUT}/check_dream.json
    !{PY} -m harness.diffusion.checks.check_reconstruction --model dream --report {OUT}/check_dream.json
    DREAM_OK = os.path.exists(f"{OUT}/check_dream.json") and json.load(open(f"{OUT}/check_dream.json"))["status"] == "PASS"
print("Dream matches the official code:", DREAM_OK)'''

DREAM_RUN = '''
if DREAM_OK and SAME_GPU:
    for s in DREAM_SEEDS:
        !{PY} -m harness.diffusion.run_reconstruction --model dream --domain math --seed {s} --items bank/pilot/pilot_bank.jsonl --out {OUT}/pilot/dream_math_s{s}.jsonl
        if RUN_PILOT_CODE:
            !{PY} -m harness.diffusion.run_reconstruction --model dream --domain code --seed {s} --items bank/pilot/pilot_bank.jsonl --out {OUT}/pilot/dream_code_s{s}.jsonl
if DREAM_OK and SAME_GPU and RUN_DEV:
    for s in (544, 545):
        !{PY} -m harness.diffusion.run_reconstruction --model dream --seed {s} --items bank/dev/dev_bank.jsonl --out {OUT}/dev/dream_s{s}.jsonl
    !python -m analysis.reconstruction samples --bank bank/dev/dev_bank.jsonl --scores {OUT}/dev --out {OUT}/dev/samples.json
if DREAM_OK and BANK and not WHOLE_BANK:
    !{PY} -m harness.diffusion.run_reconstruction --model dream --seed 544 --n-samples {N} --items {BANK} --subset {SUBSET} --out {OUT}/{NAME}/side/dream_n{N}_s544.jsonl'''

AR_RUN = '''
if RUN_AR:
    !pip install -q -r requirements.txt
    from google.colab import userdata
    from huggingface_hub import login
    login(token=userdata.get("HF_TOKEN"))
    for cfg in ("qwen_mistral", "llama_gemma"):
        for which in ("A", "B"):
            if SAME_GPU:
                !python -m harness.ar_logprob --config configs/pilot_{cfg}.json --which {which} --items bank/pilot/pilot_bank.jsonl --out {OUT}/pilot/ar_pilot_{cfg}_{which}.jsonl
            if BANK and not WHOLE_BANK:   # the full bank uses the pilot's model commits
                !python -m harness.ar_logprob --config configs/full_{cfg}.json --which {which} --items {BANK} --subset {SUBSET} --out {OUT}/{NAME}/side/ar_full_{cfg}_{which}.jsonl'''

DISCONNECT = '''
!ls -lR {OUT} | tail -40
drive.flush_and_unmount()
from google.colab import runtime
runtime.unassign()'''


def cells(repo):
    return [
        ("markdown", TITLE),
        ("code", PARAMETERS),
        ("code", SETUP.replace("REPO_URL", repo)),
        ("markdown", "## 1. LLaDA on the pilot's math answers (needed for the repair)\nLLaDA runs in its own Python 3.11 environment, as in notebook 07. The check compares our estimate with the official `get_log_likelihood` on real weights; if it fails, nothing is scored."),
        ("code", LLADA_ENV),
        ("code", LLADA_CHECK),
        ("code", LLADA_PILOT),
        ("markdown", "## 2. Does LLaDA's reconstruction score pass as the repair?\nGate A on the pilot's math answers and the repeat check between the two seeds (PLAN.md §7)."),
        ("code", REPORT),
        ("markdown", "## 3. Optional: LLaDA on the pilot's code answers, the development bank, and the math bank\nThe development bank gives the smallest sample count whose two seeds reach a rank correlation of 0.95 (PLAN.md §4). The repair always uses 128. Set `BANK` only after the final model record is merged (PLAN.md §7); the GPU type must be the one of the other full-bank scores. Scoring the whole math bank at 128 samples takes about 4.5 A100-hours, or about 14 L4-hours, for 2,100 answers (estimated)."),
        ("code", LLADA_OPTIONAL),
        ("code", BANK_SETUP),
        ("code", LLADA_BANK),
        ("markdown", "## 4. Optional: Dream (information only)\nIn its own Python 3.11 environment with Dream's pinned torch 2.5.1. The check compares with the official sampler's shifted logits."),
        ("code", DREAM_ENV),
        ("code", DREAM_RUN),
        ("markdown", "## 5. Optional: the AR arm (information only)\nQwen2.5-7B, Mistral-7B, Llama-3.1-8B and Gemma-2-9B, with the attention, precision and batch size of the verdict runs."),
        ("code", AR_RUN),
        ("markdown", "## 6. Final report, then disconnect"),
        ("code", REPORT),
        ("code", DISCONNECT),
    ]
