"""Cells of the two full-bank notebooks (PLAN.md §2), used by make_notebooks.py:
10_full_generate (GPU, plus an API section without one) and 11_full_label_split (CPU, label keeper only).
"""

GENERATE = [
 ("markdown", '''
# Full bank · generate the answers (math and code)

- **Section 1 (open models): A100.** About 2 hours for both domains.
- **Section 2 (GPT-4.1-mini by API): no GPU.** Run it in a separate CPU runtime (`RUN_OPEN_MODELS = False`, `RUN_API = True`) so it uses no A100 time. About 80 minutes and $6; it stops at $8. Needs the Colab secret `OPENROUTER_API_KEY`. Never run it in two places at once.

The question files (`bank/full/questions_math.jsonl` and `questions_code.jsonl`, from `bank.select_full`) must be in `main`. Math and code can be run together or one at a time (`DOMAINS`). Every finished chunk (200 problems; 50 for the API) is copied to Drive, so if the runtime stops, run the notebook again and it continues. No label is read or made here. Share the Drive folder with the label keeper, who runs notebook 11.'''),
 ("code", '''
DOMAINS = ["math", "code"]      # or just one of them
RUN_OPEN_MODELS = True          # section 1: needs an A100
RUN_API = False                 # section 2: needs OPENROUTER_API_KEY, no GPU
DISCONNECT = True               # release the runtime at the end
DRIVE = "/content/drive/MyDrive/cs544_full_bank"
BRANCH = "main"
DOMAINS_ARG = " ".join(DOMAINS)'''),
 ("code", '''
from google.colab import drive
drive.mount("/content/drive")
!nvidia-smi --query-gpu=name,memory.total --format=csv
%cd /content
!rm -rf /content/project && git clone -q -b {BRANCH} @REPO@ /content/project
%cd /content/project
!git log --oneline -1
!mkdir -p {DRIVE}

import datetime, os, subprocess
for d in DOMAINS:
    assert os.path.isfile(f"bank/full/questions_{d}.jsonl"), f"bank/full/questions_{d}.jsonl is missing: run bank.select_full and merge its files first"
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
with open(f"{DRIVE}/runs.txt", "a") as f:  # the commit is also in every generations_manifest.json
    f.write(f"{now} notebook 10, commit {commit}, domains {DOMAINS}, open models {RUN_OPEN_MODELS}, API {RUN_API}\\n")'''),
 ("markdown", "## 1. Open models (A100)\nPhi-4-mini, Granite-3.3-8B and OLMo-2-7B with vLLM, at the pilot's commits and library versions. Each model is loaded once for both domains and is saved to Drive before the next one starts."),
 ("code", '''
if RUN_OPEN_MODELS:
    !pip install -q vllm==0.32.0 transformers==5.19.0
    for g in ("phi4mini", "granite33", "olmo2"):
        !python -m bank.generate_full run --generator {g} --domains {DOMAINS_ARG} --save {DRIVE}'''),
 ("markdown", "## 2. GPT-4.1-mini through OpenRouter (no GPU)\nThe spend is in `api_spend.json` in the Drive folder, a new ledger that does not include the pilot's $1.19. If some answers fail, run the cell again: it asks only for the missing ones."),
 ("code", '''
if RUN_API:
    import os
    from google.colab import userdata
    os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")
    !python -m bank.generate_full run --generator gpt41mini --domains {DOMAINS_ARG} --save {DRIVE} --budget-usd 8'''),
 ("markdown", "## 3. What is done, and disconnect"),
 ("code", '''
!python -m bank.generate_full status --domains {DOMAINS_ARG} --gen {DRIVE}/gen
!head -n 2 {DRIVE}/api_spend.json'''),
 ("code", '''
from google.colab import drive, runtime
drive.flush_and_unmount()
if DISCONNECT:
    runtime.unassign()'''),
]

LABEL = [
 ("markdown", '''
# Full bank · label, pair, draw and split (label keeper only)

Runtime: **CPU** (the free one is enough). Run it only if you are the **label keeper**: the person who holds the test labels (role F in PLAN.md §2) and takes no part in scoring, in choosing pairs or in setting thresholds. Nobody else keeps a copy of anything in the private folder; people who help with the label audit or the error tags delete theirs when done.

Before you start:
- The Drive folder from notebook 10 is shared with you (add a shortcut to it in My Drive) and complete for the domains below.
- Colab secret `HF_TOKEN`, from an account with access to Llama-3.1-8B, Llama-3.2-3B, Gemma-2-9B and Mistral-7B. Their tokenizers are needed for the length filter.

Three folders:
- `SHARED`: the answers from notebook 10. Only read.
- `PRIVATE`: **your own Drive, never shared.** Labeled answers, execution logs, test labels and the error-tag sheet.
- `COMMIT`: what goes into the repository: the bank, the validation labels and the manifests.

A step that fails stops the notebook. Only counts and hashes are printed, never a test label or a result for one answer. Do not open the private files with anyone who scores, selects or sets thresholds.'''),
 ("code", '''
I_AM_THE_LABEL_KEEPER = False   # set to True only if the text above describes you
DOMAINS = ["math", "code"]      # or just one of them
VERSION = "v1"                  # a corrected bank, for example after a failed label audit, is a new version: raise this
SHARED = "/content/drive/MyDrive/cs544_full_bank"
PRIVATE = f"/content/drive/MyDrive/cs544_full_bank_private/{VERSION}"
COMMIT = f"/content/drive/MyDrive/cs544_full_bank_to_commit/{VERSION}"
WORK = f"/content/work_{VERSION}"
BOX = f"/content/sandbox_{VERSION}"
DISCONNECT = True
BRANCH = "main"
assert I_AM_THE_LABEL_KEEPER, "only the label keeper runs this notebook (PLAN.md §2)"
FOLDERS = (SHARED, PRIVATE, COMMIT)
assert sum((a + "/").startswith(b + "/") for a in FOLDERS for b in FOLDERS) == 3, "the three folders must be separate"'''),
 ("code", '''
from google.colab import drive, userdata
drive.mount("/content/drive")
%cd /content
!rm -rf /content/project && git clone -q -b {BRANCH} @REPO@ /content/project
%cd /content/project
!git log --oneline -1
!pip install -q math-verify==0.9.0 evalplus==0.3.1 transformers==5.19.0 sentencepiece tiktoken protobuf
!mkdir -p {PRIVATE} {COMMIT} {WORK} && chmod 700 {WORK}

import datetime, subprocess
from importlib.metadata import version
assert (version("math-verify"), version("evalplus")) == ("0.9.0", "0.3.1")  # the pilot's checkers
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
with open(f"{PRIVATE}/runs.txt", "a") as f:
    f.write(f"{now} notebook 11, commit {commit}, version {VERSION}, domains {DOMAINS}\\n")
STOPPED = "stopped: see the message above"'''),
 ("markdown", "## 1. Collect the answers\nStops if any generator is incomplete, or used another model commit, setting or question file."),
 ("code", '''
for d in DOMAINS:
    !python -m bank.generate_full merge --domain {d} --gen {SHARED}/gen --out {WORK}/{d}/generations.jsonl
    assert _exit_code == 0, STOPPED'''),
 ("markdown", "## 2. Math labels\nThe boxed answer and Math-Verify must agree."),
 ("code", '''
if "math" in DOMAINS:
    !python -m bank.label --questions bank/full/questions_math.jsonl --generations {WORK}/math/generations.jsonl --out {WORK}/math/labeled.jsonl
    assert _exit_code == 0, STOPPED'''),
 ("markdown", "## 3. Code labels\nEvalPlus's base and plus tests, each program twice. These are model-written programs, so they run as a user who cannot read Drive, the labels or a Hugging Face login, with an empty environment; if that user could, the cell stops before running anything. Expect several hours on a free runtime with 2 CPUs; finished checks are copied to the private folder every 5 minutes, so after a disconnect run the notebook again and it continues."),
 ("code", '''
if "code" in DOMAINS:
    import os, shutil, subprocess, threading
    CHECKS, SAVED, LOGIN = f"{BOX}/evalplus_checks.jsonl", f"{PRIVATE}/code/evalplus_checks.jsonl", "/root/.cache/huggingface"
    os.makedirs(os.path.dirname(SAVED), exist_ok=True)

    def plain(path):  # root never follows a link that the sandbox user could have made
        return os.path.isfile(path) and not os.path.islink(path)

    def save_checks():
        try:
            if plain(CHECKS) and (not os.path.isfile(SAVED) or os.path.getsize(CHECKS) > os.path.getsize(SAVED)):
                shutil.copyfile(CHECKS, SAVED)
        except OSError as exc:  # Drive hiccup: try again next time
            print("could not save the finished checks:", exc)

    save_checks()  # what an interrupted run in this runtime left
    if os.path.isdir(BOX):
        shutil.rmtree(BOX)
    os.makedirs(BOX)
    shutil.copyfile(f"{WORK}/code/generations.jsonl", f"{BOX}/generations.jsonl")
    if os.path.isfile(SAVED):
        shutil.copyfile(SAVED, CHECKS)
    !chown -R nobody {BOX}
    assert _exit_code == 0, STOPPED
    if os.path.isdir(LOGIN):
        os.chmod(LOGIN, 0o700)
    for folder in (PRIVATE, SHARED, COMMIT, WORK, LOGIN):
        assert subprocess.run(["runuser", "-u", "nobody", "--", "ls", folder], capture_output=True).returncode != 0, f"the sandbox user can read {folder}: stop"
    assert subprocess.run(["runuser", "-u", "nobody", "--", "touch", f"{BOX}/labeled.jsonl"]).returncode == 0, "the sandbox user cannot write its own folder"

    finished = threading.Event()
    def keep_saving():
        while not finished.wait(300):
            save_checks()
    saver = threading.Thread(target=keep_saving, daemon=True)
    saver.start()
    !runuser -u nobody -- env -i HOME={BOX} LANG=C.UTF-8 PATH=/usr/local/bin:/usr/bin:/bin python -m bank.label_evalplus --questions bank/full/questions_code.jsonl --generations {BOX}/generations.jsonl --out {BOX}/labeled.jsonl --progress {CHECKS}
    code_exit = _exit_code
    finished.set()
    saver.join()
    save_checks()
    assert code_exit == 0, STOPPED
    for name in ("labeled.jsonl", "labeled_manifest.json"):
        assert plain(f"{BOX}/{name}"), f"{name} is missing or is a link"
        shutil.copyfile(f"{BOX}/{name}", f"{WORK}/code/{name}")'''),
 ("markdown", "## 4. Pairs, length filter, draw and split"),
 ("code", '''
from huggingface_hub import login
login(token=userdata.get("HF_TOKEN"))
for d in DOMAINS:
    !python -m bank.make_pairs --split full --questions bank/full/questions_{d}.jsonl --generations {WORK}/{d}/generations.jsonl --labeled {WORK}/{d}/labeled.jsonl --out {WORK}/{d}
    assert _exit_code == 0, STOPPED
    !python -m bank.token_lengths {WORK}/{d}/full_bank.jsonl --out {WORK}/{d}/too_long.json
    assert _exit_code == 0, STOPPED
    !python -m bank.draw_full --domain {d} --work {WORK}/{d} --questions bank/full/questions_{d}.jsonl --generations {WORK}/{d}/generations.jsonl
    assert _exit_code == 0, STOPPED
    !python -m bank.split_full split --domain {d} --work {WORK}/{d} --out {COMMIT}/bank/full/{d} --private {PRIVATE}/{d}
    assert _exit_code == 0, STOPPED
    !mkdir -p {PRIVATE}/{d}/work && cp -r {WORK}/{d}/. {PRIVATE}/{d}/work/
    assert _exit_code == 0, STOPPED'''),
 ("markdown", "## 5. Math error tags\nWrites a sheet of 200 wrong math test answers to the private folder (PLAN.md §2). When `error_type` is filled in, set `APPLY_TAGS` to True and run the first two code cells, the second cell of this section and section 6. Do this before the bank freeze. If the sheet already exists, it is left alone."),
 ("code", '''
if "math" in DOMAINS:
    !python -m bank.split_full error-sheet --out {COMMIT}/bank/full/math --private {PRIVATE}/math'''),
 ("code", '''
APPLY_TAGS = False
if APPLY_TAGS:
    !python -m bank.split_full apply-tags --out {COMMIT}/bank/full/math --private {PRIVATE}/math
    assert _exit_code == 0, STOPPED'''),
 ("markdown", "## 6. What to commit\nThe bank freeze commits the files listed here, after the label audit has passed. The test labels stay in the private folder until they are joined, once, after the preregistration commit."),
 ("code", '''
import hashlib, shutil
from pathlib import Path
public = Path(COMMIT) / "bank/full"
for path in [p for d in DOMAINS for p in sorted(Path(SHARED).glob(f"gen/*_{d}/generations_manifest.json"))] + [Path(SHARED) / "api_spend.json"]:
    target = public / path.relative_to(SHARED)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
files = sorted(p for p in Path(COMMIT).rglob("*") if p.is_file())
private_files = [p.name for p in files if p.name not in ("bank.jsonl", "labels_validation.jsonl", "manifest.json", "generations_manifest.json", "api_spend.json")]
assert not private_files, f"must not be committed: {private_files}"
for p in files:
    print(hashlib.sha256(p.read_bytes()).hexdigest(), p.relative_to(COMMIT))
for d in DOMAINS:
    print(hashlib.sha256((Path(PRIVATE) / d / "labels_test.jsonl").read_bytes()).hexdigest(), d, "labels_test.jsonl (private, hash only)")'''),
 ("code", '''
from google.colab import drive, runtime
drive.flush_and_unmount()
if DISCONNECT:
    runtime.unassign()'''),
]


def notebooks(nb, repo):
    """(name, notebook) for both, built with make_notebooks.nb."""
    fill = lambda cells: [(kind, text.replace("@REPO@", repo)) for kind, text in cells]  # noqa: E731
    generate, label = nb(fill(GENERATE)), nb(fill(LABEL))
    label["metadata"].pop("accelerator")  # CPU runtime
    label["metadata"]["colab"].pop("gpuType")
    return [("10_full_generate", generate), ("11_full_label_split", label)]
