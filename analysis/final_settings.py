"""Settings of the final analysis: the fixed numbers, the verifiers, and every pair that is analyzed.

Two kinds of settings, kept apart on purpose:
- RULES holds every fixed number of PLAN.md §5-§7 (90% pair TAR, margin, matching limits,
  bootstrap, seed). It is code, so its hash is in the analysis manifest, and no command
  line can change it.
- The model settings file (configs/final_models.json, written from the final model record of
  the go/no-go; configs/final_models.example.json shows the format) says which row of PLAN.md
  §7's table applies: the mode, the AR verifiers left in H1, and LLaDA's score.

    from analysis.final_settings import load_settings, plan_pairs
    settings = load_settings("configs/final_models.json")
    plan_pairs(settings, "math")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path


@dataclass(frozen=True)
class Rules:
    """Every fixed number of the final analysis (PLAN.md section in the comment)."""
    pair_tar: float = 0.90              # §5 step 2: the pair accepts 90% of correct validation answers
    other_tars: tuple = (0.80, 0.95)    # §5 step 2: the other reported operating points
    margin_points: float = 3.0          # §5 decision rule: smallest effect worth caring about, points of FAR
    match_tar_points: float = 5.0       # §5 step 7: each pair's test TAR within 5 points of 90%
    match_gap_points: float = 3.0       # §5 step 7: the two pairs within 3 points of each other
    old_match_tar_points: float = 3.0   # §5 step 7 until 2026-11-21; reported beside the check, changes no verdict
    bootstrap: int = 10_000             # §5 step 8
    seed: int = 544                     # §5 step 8
    cost_ratio: float = 1.5             # §5 step 4
    gate_a_auroc: float = 0.60          # §7 gate A, reported again on validation
    gate_a_lower: float = 0.50
    gate_a_small_gap: float = 0.10
    recon_samples: int = 128            # §7 repair details
    recon_seed: int = 544
    recon_mc_group: int = 16            # the settings the reconstruction score is gated with on the pilot
    recon_micro_batch: int = 1
    power: float = 0.80                 # §6: detectable effect at 80% power, 5% two-sided
    small_group: int = 30               # breakdowns with fewer wrong answers are marked "small" (our choice)
    h2_on_trend_kappa: float = 0.10     # §8: "on the AR trend" needs a 90% interval of (kappa - trend) inside +-0.10;
                                        # otherwise inconclusive (decisions/2026-11-21-mechanism-wording.md)

    @property
    def tars(self) -> tuple:
        return tuple(sorted({self.pair_tar, *self.other_tars}))


RULES = Rules()
H1_DOMAIN = "math"           # §5: H1 is math only
DOMAINS = ("math", "code")
TRADEOFF_TARS = tuple(round(0.70 + 0.01 * i, 2) for i in range(30))  # validation thresholds for the trade-off curves


@dataclass(frozen=True)
class Verifier:
    name: str
    family: str
    kind: str           # "AR" or "diffusion"
    role: str           # "primary", "small" or "extra"
    keys: tuple         # model ids / short keys that name it in score files
    attention: str = "sdpa"     # AR models: the attention they were scored with on the pilot (§13 row 35)


VERIFIERS = {v.name: v for v in (
    Verifier("Qwen2.5-7B", "qwen", "AR", "primary", ("Qwen/Qwen2.5-7B-Instruct",)),
    Verifier("Llama-3.1-8B", "llama", "AR", "primary", ("meta-llama/Llama-3.1-8B-Instruct",)),
    Verifier("Gemma-2-9B", "gemma", "AR", "primary", ("google/gemma-2-9b-it",), "eager"),
    Verifier("Mistral-7B", "mistral", "AR", "primary", ("mistralai/Mistral-7B-Instruct-v0.3",)),
    Verifier("Llama-3.2-3B", "llama", "AR", "small", ("meta-llama/Llama-3.2-3B-Instruct",)),
    Verifier("Qwen2.5-1.5B", "qwen", "AR", "small", ("Qwen/Qwen2.5-1.5B-Instruct",)),
    Verifier("Qwen3-8B", "qwen", "AR", "extra", ("Qwen/Qwen3-8B",)),
    Verifier("LLaDA-8B", "llada", "diffusion", "primary", ("llada", "GSAI-ML/LLaDA-8B-Instruct")),
    Verifier("Dream-7B", "dream", "diffusion", "primary", ("dream", "Dream-org/Dream-v0-Instruct-7B")),
)}
BY_KEY = {key: v.name for v in VERIFIERS.values() for key in v.keys}
ANSWER_SLOTS, ANSWER_PREFIX = 1, "none"     # LLaDA and Dream, set on the development bank (§4) and used on the pilot
DIFFUSION, SMALL_REFERENCE, STRONG_BASELINE = "LLaDA-8B", "Llama-3.2-3B", "Qwen3-8B"
E1_ANCHOR, E1_DIFFUSION = "Qwen2.5-7B", "Dream-7B"
H1_AR_POOL = ("Gemma-2-9B", "Llama-3.1-8B", "Qwen2.5-7B")           # §7: the pool can only shrink
AR_FAMILY = ("Gemma-2-9B", "Llama-3.1-8B", "Mistral-7B", "Qwen2.5-7B")   # §7 pivot: the four AR verifiers
TREND_AR = AR_FAMILY + ("Llama-3.2-3B", "Qwen2.5-1.5B")             # §5 H2: competence trend

# Score columns. T1/T2: verdict scores. avg: mean of T1 and T2. recon: reconstruction score (repair only).
COLUMN_LABEL = {"T1": "", "T2": " (T2)", "avg": " (T1/T2 mean)", "recon": " (reconstruction)"}


@dataclass(frozen=True)
class Member:
    """One verifier's score inside a pair."""
    verifier: str
    column: str = "T1"

    @property
    def label(self) -> str:
        return self.verifier + COLUMN_LABEL[self.column]


@dataclass(frozen=True)
class Pair:
    a: Member
    b: Member
    kind: str           # "AR+AR", "AR+diffusion" or "pseudo"
    roles: tuple        # h1_ar, h1_mixed, e1, h2_trend, pseudo, qwen3, ar_family, template_avg, background

    @property
    def name(self) -> str:
        return f"{self.a.label} + {self.b.label}"

    @property
    def members(self) -> tuple:
        return self.a, self.b


def make_pair(x: Member, y: Member, roles) -> Pair:
    """A pair of two members in alphabetical order, so a pair has one name whatever the input order."""
    a, b = sorted((x, y), key=lambda m: m.label)
    if a.verifier == b.verifier:
        kind = "pseudo"
    else:
        kinds = {VERIFIERS[a.verifier].kind, VERIFIERS[b.verifier].kind}
        if kinds == {"diffusion"}:
            raise ValueError(f"{a.label} + {b.label}: two diffusion verifiers are never paired in this plan")
        kind = "AR+AR" if kinds == {"AR"} else "AR+diffusion"
    return Pair(a, b, kind, tuple(roles))


# ---------------------------------------------------------------- model settings

MODES = ("go", "repair", "pivot", "stopped")


class SettingsError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    mode: str               # go | repair | pivot | stopped (PLAN.md §7's table; rows combine with a smaller pool)
    h1_ar: tuple            # AR verifiers left in H1
    llada_score: str | None  # "verdict", "reconstruction" or None (pivot, stopped)
    dropped: dict           # verifier -> reason
    reason: str
    record: str             # path of the final model record in decisions/
    qwen3_scored: bool      # Qwen3-8B scores exist (required when fewer than three AR verifiers are left)

    @property
    def runs_h1(self) -> bool:
        return self.mode in ("go", "repair")

    @property
    def baseline_ar(self) -> tuple:
        """AR verifiers of the AR+AR baseline: the pool, plus Qwen3-8B when fewer than three are left and it passed
        its own checks (§3)."""
        return self.h1_ar + ((STRONG_BASELINE,) if len(self.h1_ar) < 3 and STRONG_BASELINE not in self.dropped else ())

    @property
    def test_name(self) -> str:
        return ("AR verdict versus diffusion reconstruction score" if self.mode == "repair"
                else "AR+diffusion pair versus AR+AR pair (verdict scores)")

    def llada_column(self, domain: str) -> str:
        """LLaDA's score in a domain. The repair is LLaDA on math only (§7); code keeps the verdict score."""
        return "recon" if self.llada_score == "reconstruction" and domain == H1_DOMAIN else "T1"

    def to_dict(self) -> dict:
        return asdict(self)


def cross_family(a: str, b: str) -> bool:
    return VERIFIERS[a].family != VERIFIERS[b].family


def cross_family_pairs(names) -> list[tuple]:
    return [(a, b) for a, b in combinations(sorted(names), 2) if cross_family(a, b)]


def parse_settings(d: dict) -> Settings:
    """Check a model settings dict against PLAN.md §3 and §7 and return it. Anything the go/no-go
    cannot produce is refused, so the file cannot quietly change the final test."""
    allowed = {"schema", "mode", "h1_ar", "llada_score", "dropped", "reason", "record", "qwen3_scored", "note"}
    if set(d) - allowed:
        raise SettingsError(f"unknown settings keys {sorted(set(d) - allowed)}; the fixed numbers live in "
                            "analysis/final_settings.py and cannot be set here")
    if d.get("schema") != 1:
        raise SettingsError("schema must be 1")
    mode = d.get("mode")
    if mode not in MODES:
        raise SettingsError(f"mode must be one of {MODES}, not {mode!r}")
    h1_ar = tuple(sorted(d.get("h1_ar", ())))
    if len(set(h1_ar)) != len(h1_ar) or not set(h1_ar) <= set(H1_AR_POOL):
        raise SettingsError(f"h1_ar must be a subset of {H1_AR_POOL} without repeats: the pool can only shrink "
                            f"(PLAN.md §7); got {list(h1_ar)}")
    dropped = dict(d.get("dropped") or {})
    left = set(H1_AR_POOL) - set(h1_ar)
    if STRONG_BASELINE in dropped and len(h1_ar) < 3:       # it joins only if it passes tests 2-3 (§3)
        left.add(STRONG_BASELINE)
    if mode == "pivot":
        pool = h1_ar + ((STRONG_BASELINE,) if len(h1_ar) < 3 and STRONG_BASELINE not in dropped else ())
        if DIFFUSION in dropped:
            left.add(DIFFUSION)
        elif h1_ar and cross_family_pairs(pool):
            raise SettingsError("a pivot needs LLaDA in dropped, or an AR pool without a cross-family pair (PLAN.md §7)")
    if mode != "stopped" and (set(dropped) != left
                              or not all(isinstance(r, str) and r.strip() for r in dropped.values())):
        raise SettingsError(f"dropped must give a reason for exactly the verifiers that left: {sorted(left)}, "
                            f"got {sorted(dropped)}")
    score = d.get("llada_score")
    expected = {"go": "verdict", "repair": "reconstruction"}.get(mode)
    if score != expected:
        raise SettingsError(f"mode {mode!r} needs llada_score {expected!r}, not {score!r}")
    reason, record = d.get("reason", ""), d.get("record", "")
    if not record:
        raise SettingsError("record must name the final model record in decisions/")
    if mode in ("repair", "pivot", "stopped") and not reason:
        raise SettingsError(f"mode {mode!r} needs a reason")
    qwen3 = bool(d.get("qwen3_scored", False))
    s = Settings(mode, h1_ar, score, dropped, reason, record, qwen3)
    if s.runs_h1:
        if not cross_family_pairs(s.baseline_ar):
            raise SettingsError("no cross-family AR+AR pair can be formed from "
                                f"{list(s.baseline_ar)}: H1 is not run (PLAN.md §3), so mode must be 'pivot'")
        if (STRONG_BASELINE in s.baseline_ar) and not qwen3:
            raise SettingsError("fewer than three AR verifiers are left, so Qwen3-8B joins the AR+AR baseline "
                                "(PLAN.md §3): score it and set qwen3_scored to true, or give the reason it failed its "
                                "reference or repeat-run check in dropped")
    if STRONG_BASELINE in dropped and qwen3:
        raise SettingsError("Qwen3-8B is in dropped, so qwen3_scored must be false")
    return s


def load_settings(path) -> Settings:
    return parse_settings(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------- which scores and pairs are analyzed

def needed_scores(settings: Settings, domain: str) -> list[tuple]:
    """(verifier, column) of every score file the analysis needs in a domain. T1 and T2 of every
    verifier scored on the full bank (NEXT_STEPS C and D), Qwen3-8B when scored, and LLaDA's
    reconstruction score on math in repair mode."""
    names = [v for v in VERIFIERS if v != STRONG_BASELINE or settings.qwen3_scored]
    out = [(v, t) for v in names for t in ("T1", "T2")]
    if settings.llada_column(domain) == "recon":
        out.append((DIFFUSION, "recon"))
    return out


def plan_pairs(settings: Settings, domain: str) -> list[Pair]:
    """Every pair analyzed anywhere in the plan, in one domain, each with the roles it plays.

    The same two members never appear twice: roles of a repeated pair are merged."""
    found: dict[tuple, list] = {}

    def add(x: Member, y: Member, *roles):
        key = tuple(sorted((x, y), key=lambda m: m.label))
        found.setdefault(key, [])
        found[key] += [r for r in roles if r not in found[key]]

    def verdict(name):
        return Member(name, "T1")

    llada = Member(DIFFUSION, settings.llada_column(domain))
    if settings.runs_h1:
        for a, b in cross_family_pairs(settings.baseline_ar):
            add(verdict(a), verdict(b), "h1_ar")
        # In repair mode code's LLaDA pairs use the verdict score, which is in no final test
        role = "background" if settings.mode == "repair" and llada.column != "recon" else "h1_mixed"
        for a in settings.h1_ar:
            add(verdict(a), llada, role)
    elif settings.mode == "pivot":
        for a, b in cross_family_pairs(AR_FAMILY):
            add(verdict(a), verdict(b), "ar_family")

    # E1: Qwen2.5-7B with Dream, with LLaDA and with each other AR model (verdict scores, like for like)
    add(verdict(E1_ANCHOR), verdict(E1_DIFFUSION), "e1")
    add(verdict(E1_ANCHOR), verdict(DIFFUSION), "e1")
    for name in AR_FAMILY:
        if name != E1_ANCHOR:
            add(verdict(E1_ANCHOR), verdict(name), "e1")

    # H2 trend: cross-family AR pairs including the two small AR models, plus Qwen3-8B when it is in the pool
    trend = TREND_AR + ((STRONG_BASELINE,) if settings.runs_h1 and STRONG_BASELINE in settings.baseline_ar else ())
    for a, b in cross_family_pairs(trend):
        add(verdict(a), verdict(b), "h2_trend")

    # Qwen3-8B with every AR and diffusion partner of the final test (strong-baseline check)
    if settings.qwen3_scored:
        for name in settings.h1_ar if settings.runs_h1 else AR_FAMILY:
            if cross_family(name, STRONG_BASELINE):
                add(verdict(name), verdict(STRONG_BASELINE), "qwen3")
        add(llada, verdict(STRONG_BASELINE), "qwen3")

    # Same-model pseudo-pairs: T1 with T2
    for name in VERIFIERS:
        if name != STRONG_BASELINE or settings.qwen3_scored:
            add(Member(name, "T1"), Member(name, "T2"), "pseudo")

    # Template-averaged versions of the final test's pairs, for the budget-matched comparison (§5 step 4)
    if settings.runs_h1:
        def averaged(m: Member) -> Member:
            return m if m.column == "recon" else Member(m.verifier, "avg")
        for key, roles in list(found.items()):
            if {"h1_ar", "h1_mixed"} & set(roles):
                add(averaged(key[0]), averaged(key[1]), "template_avg")

    # Background rows, in no test: Dream and LLaDA verdict scores with each AR model not already paired above
    for name in AR_FAMILY:
        for diffusion in (E1_DIFFUSION, DIFFUSION):
            key = tuple(sorted((verdict(name), verdict(diffusion)), key=lambda m: m.label))
            if key not in found:
                add(*key, "background")
    return [make_pair(x, y, roles) for (x, y), roles in found.items()]


def single_members(settings: Settings, domain: str) -> list[Member]:
    """Every single score that gets its own 90% threshold."""
    out = [Member(v, c) for v, c in needed_scores(settings, domain)]
    names = sorted({m.verifier for m in out})
    return out + [Member(v, "avg") for v in names]


def pseudo_pair_name(verifier: str) -> str:
    """Name of a verifier's same-model pseudo-pair (T1 with T2)."""
    return make_pair(Member(verifier, "T1"), Member(verifier, "T2"), ()).name


def final_test_members(settings: Settings, domain: str = H1_DOMAIN) -> list[Member]:
    """The verifiers that take part in the final test."""
    return [Member(v, "T1") for v in settings.baseline_ar] + [Member(DIFFUSION, settings.llada_column(domain))]


def primary_members(settings: Settings, domain: str = H1_DOMAIN) -> list[Member]:
    """Every scored verifier with its primary score (T1, or LLaDA's score in the final test): the candidates for
    "best single verifier", which PLAN §5 does not limit to the final test's verifiers."""
    names = sorted({v for v, _ in needed_scores(settings, domain)})
    return [Member(v, settings.llada_column(domain) if v == DIFFUSION else "T1") for v in names]


def same_verifiers(settings: Settings, pair: Pair, domain: str) -> Pair | None:
    """The pair of the same two verifiers in another domain, with the scores they have there (in repair mode LLaDA
    has its reconstruction score on math and its verdict score on code)."""
    wanted = {m.verifier for m in pair.members}
    for p in plan_pairs(settings, domain):
        primary = all(m.column == (settings.llada_column(domain) if m.verifier == DIFFUSION else "T1") for m in p.members)
        if primary and {m.verifier for m in p.members} == wanted:
            return p
    return None


def with_role(pairs: list[Pair], role: str) -> list[Pair]:
    return [p for p in pairs if role in p.roles]
