"""Cells of notebook 09 (step 0 of the go/no-go): the answer-swap test and the AR repeat runs.

make_notebooks.py imports this module and writes 09_step0.ipynb. Edit the cells here.
"""

TITLE = '''
# Step 0 · answer-swap test (test 4) and AR repeat runs

Runtime: **A100**. *Runtime → Run all*, and allow Google Drive when asked. Add a Hugging Face token as the Colab secret `HF_TOKEN`, with access to Llama, Gemma and Mistral. About an hour, mostly model loading.

What it runs (PLAN.md §4 and §7), all with T1 and batch size 1:
1. **Part 1, what the go/no-go needs:** the 400 swap rows with Llama-3.1-8B, Gemma-2-9B, Qwen2.5-7B and LLaDA-8B, and the 500 pilot answers twice with the three AR models. (Mistral comes along in Qwen's config.)
2. **Part 2, the other verifiers:** Llama-3.2-3B, Qwen2.5-1.5B and Dream.

**Every step is saved to Google Drive** (`MyDrive/cs544_step0/`). After a disconnect, run all cells again: finished steps are skipped, and the code stays on the commit that made the first saved scores. The last cell disconnects the runtime.

**To start over** (a fix was merged, or the swap items or the pilot bank changed), delete `MyDrive/cs544_step0/` first.

No label is read while scoring. The analysis reads only which row of a swap pair is the original (`swap_key.jsonl`).'''

SETUP = '''
BRANCH = "main"
import hashlib, json, shutil, subprocess
from pathlib import Path
GPU = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
print(GPU)
assert "A100" in GPU, "Pick an A100: Runtime > Change runtime type"
from google.colab import drive, userdata
from huggingface_hub import login
drive.mount("/content/drive")
login(token=userdata.get("HF_TOKEN"))
DRIVE = Path("/content/drive/MyDrive/cs544_step0")
SAVED = DRIVE / "step0"
SAVED.mkdir(parents=True, exist_ok=True)
if not Path("/content/project").exists():
    !git clone -q -b {BRANCH} %s /content/project
%%cd /content/project
SWAP, PILOT = "bank/pilot/swap/swap_items.jsonl", "bank/pilot/pilot_bank.jsonl"
COMMIT = (SAVED / "commit.txt").read_text().strip() if (SAVED / "commit.txt").exists() else "origin/" + BRANCH
newer = subprocess.check_output(["git", "diff", "--name-only", COMMIT, "origin/" + BRANCH, "--", SWAP, PILOT], text=True)
assert not newer, "these changed since the saved run; delete MyDrive/cs544_step0 to start over: " + newer
!git checkout -q {COMMIT}   # a resumed run keeps the code of the first saved scores
HEAD = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert COMMIT in ("origin/" + BRANCH, HEAD), "could not go back to the saved run's commit " + COMMIT
COMMIT = HEAD
print("commit", COMMIT)
!pip install -q -r requirements.txt'''

HELPERS = '''
OUT = Path("results/step0")
SWAP_HASH, PILOT_HASH = (hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (SWAP, PILOT))
N_SWAP = sum(1 for _ in open(SWAP, encoding="utf-8"))
assert SWAP_HASH == json.loads(Path("bank/pilot/swap/swap_manifest.json").read_text())["outputs"]["swap_items.jsonl"]
shutil.rmtree(OUT, ignore_errors=True)   # Drive is the record, not results that came with the clone
shutil.copytree(SAVED, OUT, ignore=shutil.ignore_patterns("*.part"))
(OUT / "commit.txt").write_text(COMMIT + "\\n")

def save(step):
    """Copy one finished step to Drive. It gets its real name only once the copy is complete."""
    src, dst = OUT / step, SAVED / step
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    def remove(path):
        shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)
    remove(part)
    (shutil.copytree if src.is_dir() else shutil.copy2)(src, part)
    remove(dst)
    part.rename(dst)
    if step.startswith(("swap/", "rerun_")):   # the first saved scores fix the commit for later runs
        (SAVED / "commit.txt").write_text(COMMIT + "\\n")
    print("saved to Drive:", step)

def ar_done(step):
    path = OUT / step / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    return manifest.get("status") == "completed" and manifest["input_sha256"] == (SWAP_HASH if step.startswith("swap/") else PILOT_HASH)

def diffusion_done(step):
    path = OUT / step
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return len(rows) == N_SWAP and all(r["status"] == "ok" and r["bank_hash"] == SWAP_HASH for r in rows)

print(sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()))'''

LENGTHS = '''
if not ((SAVED / "swap_token_lengths.json").exists() and (SAVED / "commit.txt").exists()):   # saved only if it passed
    (OUT / "swap_token_lengths.json").unlink(missing_ok=True)
    !python -m bank.token_lengths {SWAP} --out {OUT}/swap_token_lengths.json
assert (OUT / "swap_token_lengths.json").exists(), "no lengths were written; see the message above (usually a tokenizer this token cannot open)"
lengths = json.loads((OUT / "swap_token_lengths.json").read_text())
assert not lengths["tokenizers_failed"], "give the token access to these, then run this cell again: " + str(sorted(lengths["tokenizers_failed"]))
assert not lengths["too_long"], "a swap prompt is too long; stop and report it"
save("swap_token_lengths.json")'''

AR_LOOP = '''
for cfg in (%s):
    step = f"swap/{cfg}"
    if not ar_done(step):
        shutil.rmtree(OUT / step, ignore_errors=True)
        !python -m harness.run --config configs/{cfg}.json --input {SWAP} --template T1 --output-dir {OUT}/{step} --device cuda:0 --dtype bf16 --no-load-in-4bit
        assert ar_done(step), step + " did not finish; read the messages above, then run this cell again"
        save(step)
    runs = [f"rerun_1/{cfg}", f"rerun_2/{cfg}"]
    if not all(ar_done(step) for step in runs):   # both in one session, so both are redone together
        for step in runs:
            shutil.rmtree(OUT / step, ignore_errors=True)
            !python -m harness.run --config configs/{cfg}.json --input {PILOT} --template T1 --output-dir {OUT}/{step} --device cuda:0 --dtype bf16 --no-load-in-4bit
            assert ar_done(step), step + " did not finish; read the messages above, then run this cell again"
        for step in runs:
            save(step)
    !rm -rf /root/.cache/huggingface/hub/models--*'''

DIFFUSION = '''
step = "swap/{model}_T1_n1.jsonl"
if not diffusion_done(step):
    for old in (step, step + ".manifest.jsonl"):   # never continue a half-written file: a swap and its original belong to one run
        (OUT / old).unlink(missing_ok=True)
    !pip install -q uv
    !uv venv -q --clear --python 3.11 /content/env_{model}
    !uv pip install -q --python /content/env_{model}/bin/python {torch} -r harness/diffusion/requirements/{model}.txt
    PY = "/content/env_{model}/bin/python"
    !{{PY}} -c "import sys, torch, transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__, torch.cuda.is_available())"
    !{{PY}} -m harness.diffusion.run_score --model {model} --n-slots 1 --batch-size 1 --template T1 --items {{SWAP}} --bank-hash {{SWAP_HASH}} --out {{OUT}}/{{step}}
    assert diffusion_done(step), step + " is not complete; read the messages above, then run this cell again"
    save(step + ".manifest.jsonl")
    save(step)
    !rm -rf /root/.cache/huggingface/hub/models--*'''

ANALYSIS = '''
for folder in ("swap", "rerun_1", "rerun_2"):
    (OUT / folder).mkdir(exist_ok=True)
(OUT / "step0.json").unlink(missing_ok=True)
!python -m analysis.step0 --swap-scores {OUT}/swap --rerun {OUT}/rerun_1 {OUT}/rerun_2 --out {OUT}
assert (OUT / "step0.json").exists(), "the analysis failed; read the message above (the scores are saved)"
save("step0.json")
save("step0.md")'''

FINISH = '''
!zip -qr /content/step0_results.zip results/step0
shutil.copy2("/content/step0_results.zip", DRIVE / "step0_results.zip")
drive.flush_and_unmount()
from google.colab import runtime
runtime.unassign()'''


def cells(repo):
    return [
        ("markdown", TITLE),
        ("code", SETUP % repo),
        ("code", HELPERS),
        ("markdown", "## Prompt lengths\nEvery swap prompt against the 1,760-token limit, in the tokenizers `bank/token_lengths.py` lists. The cell stops, before anything is scored, if a tokenizer does not load: give the token access and run it again. The AR scorer also refuses a prompt that is too long."),
        ("code", LENGTHS),
        ("markdown", "## Part 1a. Llama-3.1-8B and Gemma-2-9B, then Qwen2.5-7B and Mistral\nFor each config: the swap rows, then the pilot bank twice (a fresh model load each time)."),
        ("code", AR_LOOP % '"pilot_llama_gemma", "pilot_qwen_mistral"'),
        ("markdown", "## Part 1b. LLaDA-8B, in its own Python 3.11 environment\nColab's Python cannot install LLaDA's pins (as in notebooks 02 and 07). Expect 3.11, 2.6.0 and 4.38.2."),
        ("code", DIFFUSION.format(model="llada", torch="torch==2.6.0")),
        ("markdown", "## Part 1c. Results so far: all the go/no-go needs from this notebook"),
        ("code", ANALYSIS),
        ("markdown", "## Part 2a. Llama-3.2-3B and Qwen2.5-1.5B"),
        ("code", AR_LOOP % '"pilot_small",'),
        ("markdown", "## Part 2b. Dream-7B, in its own Python 3.11 environment\nExpect 3.11, 2.5.1 and 4.46.2."),
        ("code", DIFFUSION.format(model="dream", torch="torch==2.5.1")),
        ("markdown", "## Results for every verifier"),
        ("code", ANALYSIS),
        ("markdown", "## Save everything and disconnect\nThe zip is `MyDrive/cs544_step0/step0_results.zip`; unzip it into the repo (`results/step0/`). The last line gives the GPU back."),
        ("code", FINISH),
    ]
