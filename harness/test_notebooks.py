"""Offline tests for the Colab notebooks: up to date, every shell line expands, every flag exists."""
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))
NAMESPACE = {"BRANCH": "main", "PY": "/content/env/bin/python", "OUT": "results/stage3/x", "N": 1, "n": 8, "p": "think_nl", "PREFIX": "think_nl", "FILES": "a.jsonl b.jsonl",
             "DEV_HASH": "d" * 64, "SANITY_HASH": "s" * 64, "PILOT_HASH": "h" * 64,
             "d": "math", "t": "T1", "cfg": "pilot_small"}


def shell_lines():
    for path in NOTEBOOKS:
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            if cell["cell_type"] == "code":
                for line in "".join(cell["source"]).splitlines():
                    if line.strip().startswith("!"):
                        yield path.name, line.strip()[1:]


def module_source(module):
    path = Path(*module.split("."))
    text = path.with_suffix(".py").read_text(encoding="utf-8")
    if module.startswith("harness.diffusion.checks."):  # shared flags live in checks/common.py
        text += Path("harness/diffusion/checks/common.py").read_text(encoding="utf-8")
    return text


class NotebookTests(unittest.TestCase):
    def test_notebooks_match_their_generator(self):
        with tempfile.TemporaryDirectory() as out:
            subprocess.run([sys.executable, "notebooks/make_notebooks.py", out], check=True, capture_output=True)
            for path in NOTEBOOKS:
                self.assertEqual((Path(out) / path.name).read_bytes(), path.read_bytes(), path.name)

    def test_every_shell_line_expands_fully(self):
        try:
            from IPython.core.interactiveshell import InteractiveShell
        except ImportError:
            self.skipTest("IPython not installed")
        shell = InteractiveShell.instance()
        shell.user_ns.update(NAMESPACE)
        for name, line in shell_lines():
            expanded = shell.var_expand(line)
            self.assertNotRegex(expanded, r"[{}$]", f"{name}: {line}")

    def test_every_flag_exists_in_the_module_it_calls(self):
        for name, line in shell_lines():
            match = re.match(r"(?:python|\{PY\}) -m (\S+)(.*)", line)
            if not match or not match.group(1).startswith("harness."):
                continue
            source = module_source(match.group(1))
            for flag in re.findall(r"(--[a-z0-9-]+)", match.group(2)):
                if flag.startswith("--no-"):  # BooleanOptionalAction
                    flag = "--" + flag[5:]
                text = source.replace("'", '"')
                # flags are defined as "--x", or as "x" inside a loop that adds the dashes
                self.assertTrue(f'"{flag}"' in text or f'"{flag[2:]}"' in text, f"{name}: {flag} not in {match.group(1)}")


if __name__ == "__main__":
    unittest.main()
