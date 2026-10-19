"""Diffusion verifier scorers: LLaDA, iLLaDA and Dream (role D, PLAN.md §4)."""
from .interface import SCORE_ROW_FIELDS, score_row, t1_message
from .scorer import SPECS, Verdict, load_scorer, verdict_from_logits, verdict_tokens
