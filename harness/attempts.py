"""The adversarial arm's attempts in one registry ([an unreleased plan]).

The session_006 and session_007 code hard-coded training seeds 2, 3 and 4 and the attempt names "A1" and "A2". The
session_008 round runs both arms again on fresh seeds 5, 6 and 7, so every module that reads an attempt
(harness/s6_readout.py, harness/s6_rules.py, harness/s7_reading.py) takes it from here. Each attempt names its standard
arm, its adversarial runs with their seeds, its data (the dose record in data/session_006/build/datasets.json, "A1" or
"A2"), the attempt it escalates to if its gate fails and the attempt it reruns, and, through its standard arm, its
results root, cache root and seal folder.

    A1      std_s2-std_s4  adv_s2-adv_s4        data A1  escalates to A2      results/session_006
    A2      std_s2-std_s4  adv_s2_a2-adv_s4_a2  data A2  reruns A1            results/session_006
    rep     std_s5-std_s7  adv_s5-adv_s7        data A1  escalates to rep_dd  results/session_008
    rep_dd  std_s5-std_s7  adv_s5_dd-adv_s7_dd  data A2  reruns rep           results/session_008
    lr1e3   std_s5_lr1e3   adv_s5_lr1e3         data A1                       results/session_010

In prose, "rep" is the fresh-seed replication and "rep_dd" its double-dose rerun; "A2" stays the name of the session_006
double-dose rerun only. Every run of a session_008 attempt takes step 0 from the committed untrained reading under
results/session_006/untrained/, read-only (untrained_root).

In the session_010 round, "lr1e3" is the stronger-training pair at learning rate 1e-3 ([an unreleased plan]), one
standard run (arm "s10_lr1e3") and one adversarial run, escalating to nothing and rerunning nothing; it too takes step 0
from results/session_006/untrained/. Its one-run arm is kept out of seed_sets(), so that no rule's accepted seed sets widen.

The guard: nothing writes under an adversarial run's folders once they are sealed (refuse_sealed_write). The session_006
ones are always protected, results/session_006/adv_* and .cache/session_006/adv_*, since A1's committed seal hashes them; a
session_008 attempt's are protected once its gate verdict file exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS6, CACHE6 = ROOT / "results" / "session_006", ROOT / ".cache" / "session_006"
RESULTS8, CACHE8 = ROOT / "results" / "session_008", ROOT / ".cache" / "session_008"
RESULTS10, CACHE10 = ROOT / "results" / "session_010", ROOT / ".cache" / "session_010"
UNTRAINED = "untrained"


@dataclass(frozen=True)
class StandardArm:
    """Three standard runs, each with its training seed, and the folders they are read into. in_seed_sets:
    False keeps the arm's seeds out of seed_sets(), the seed sets a standard-arm table may carry (harness/s6_rules.py)."""

    name: str
    runs: tuple[tuple[str, int], ...]
    results: Path
    cache: Path
    in_seed_sets: bool = True

    @property
    def run_names(self) -> list[str]:
        return [r for r, _ in self.runs]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(k for _, k in self.runs)

    def table_runs(self) -> dict:
        """The standard runs as the rules script's table names them: {run: {"arm": "std", "seed"}}."""
        return {r: {"arm": "std", "seed": k} for r, k in self.runs}


@dataclass(frozen=True)
class Attempt:
    name: str
    arm: str  # the key of its standard arm in ARMS
    adv: tuple[tuple[str, int], ...]
    data: str  # "A1" or "A2": the dose record in data/session_006/build/datasets.json, and its dataset
    escalates_to: str | None  # the attempt run if this one's gate fails (further-run rule 1); None for a rerun
    rerun_of: str | None  # the first attempt this one reruns at double the dose; None for a first attempt

    @property
    def standard(self) -> StandardArm:
        return ARMS[self.arm]

    @property
    def results(self) -> Path:
        return self.standard.results

    @property
    def cache(self) -> Path:
        return self.standard.cache

    @property
    def seal_dir(self) -> Path:
        return self.results / "seal"

    @property
    def out_dir(self) -> Path:
        """Where its verdicts (or description) and readings are written: <results>/adversarial_arm/<name>/."""
        return self.results / "adversarial_arm" / self.name

    @property
    def adv_names(self) -> list[str]:
        return [r for r, _ in self.adv]

    @property
    def seeds(self) -> tuple[int, ...]:
        return self.standard.seeds

    def table_runs(self) -> dict:
        """The three standard runs and the attempt's three, as the rules script's table names them."""
        return {**self.standard.table_runs(), **{r: {"arm": "adv", "seed": k} for r, k in self.adv}}


ARMS = {
    "s6": StandardArm("s6", (("std_s2", 2), ("std_s3", 3), ("std_s4", 4)), RESULTS6, CACHE6),
    "s8": StandardArm("s8", (("std_s5", 5), ("std_s6", 6), ("std_s7", 7)), RESULTS8, CACHE8),
    "s10_lr1e3": StandardArm("s10_lr1e3", (("std_s5_lr1e3", 5),), RESULTS10, CACHE10, in_seed_sets=False),  # session_010
}
ATTEMPTS = {
    "A1": Attempt("A1", "s6", (("adv_s2", 2), ("adv_s3", 3), ("adv_s4", 4)), "A1", "A2", None),
    "A2": Attempt("A2", "s6", (("adv_s2_a2", 2), ("adv_s3_a2", 3), ("adv_s4_a2", 4)), "A2", None, "A1"),
    "rep": Attempt("rep", "s8", (("adv_s5", 5), ("adv_s6", 6), ("adv_s7", 7)), "A1", "rep_dd", None),
    "rep_dd": Attempt("rep_dd", "s8", (("adv_s5_dd", 5), ("adv_s6_dd", 6), ("adv_s7_dd", 7)), "A2", None, "rep"),
    "lr1e3": Attempt("lr1e3", "s10_lr1e3", (("adv_s5_lr1e3", 5),), "A1", None, None),  # session_010: the stronger-training pair
}
NAMES = tuple(ATTEMPTS)
# The committed untrained step 0 each results root takes its step-0 readings from: results/session_008 reads
# results/session_006's, read-only, and so does results/session_010. A root not listed (results/session_006, or a test's
# temporary folder) holds its own.
UNTRAINED_FROM = {RESULTS8: RESULTS6, RESULTS10: RESULTS6}


def attempt(name: str) -> Attempt:
    if name not in ATTEMPTS:
        raise ValueError(f"attempt {name!r}: one of {', '.join(NAMES)}")
    return ATTEMPTS[name]


def arm(name: str) -> StandardArm:
    if name not in ARMS:
        raise ValueError(f"standard arm {name!r}: one of {', '.join(ARMS)}")
    return ARMS[name]


def attempt_of(run: str) -> str | None:
    """The attempt an adversarial run belongs to; None for any other run. An adv_* run in no attempt is refused, so that
    no adversarial run can pass the seal's checks unregistered."""
    for a in ATTEMPTS.values():
        if run in a.adv_names:
            return a.name
    if run.startswith("adv_"):
        raise ValueError(f"{run} is an adversarial run of no registered attempt ({', '.join(NAMES)})")
    return None


def arm_of(run: str) -> StandardArm | None:
    """The standard arm a standard run belongs to; None for any other run."""
    for s in ARMS.values():
        if run in s.run_names:
            return s
    return None


def roots_of(run: str) -> tuple[Path, Path]:
    """The results and cache roots of a run: its attempt's or its standard arm's; results/session_006's for any other run
    (the untrained step 0, and session_004 and session_005 names read through the session_006 code)."""
    att = attempt_of(run)
    if att is not None:
        return ATTEMPTS[att].results, ATTEMPTS[att].cache
    s = arm_of(run)
    if s is not None:
        return s.results, s.cache
    return RESULTS6, CACHE6


def untrained_root(results: Path) -> Path:
    """The results root whose untrained/ folder holds the step 0 that runs under `results` read."""
    r = Path(results).resolve()
    for k, v in UNTRAINED_FROM.items():
        if Path(k).resolve() == r:
            return Path(v)
    return Path(results)


def cache_of_results(results: Path) -> Path:
    """The cache root that goes with a results root: a standard arm's own; the session_006 one for any other root (the
    default the session_006 code always used)."""
    r = Path(results).resolve()
    for s in ARMS.values():
        if s.results.resolve() == r:
            return s.cache
    return CACHE6


def seed_sets() -> set[tuple[int, ...]]:
    """The training seeds of every registered standard arm (those with in_seed_sets, so not s10_lr1e3's)."""
    return {s.seeds for s in ARMS.values() if s.in_seed_sets}


# ---------------------------------------------------------------------------
# The guard: no write under a sealed adversarial run's folders
# ---------------------------------------------------------------------------


class SealedWrite(PermissionError):
    """A write under an adversarial run's results or cache folder after its seal."""


def protected_folders() -> list[Path]:
    """Every adversarial run folder nothing may write under: all of the session_006 ones (results/session_006/adv_*
    and .cache/session_006/adv_*, which A1's committed seal hashes; A2 cannot run, as A1 fired no escalation), and each
    session_008 attempt's once its gate verdict file exists."""
    out = []
    for a in ATTEMPTS.values():
        if a.results == RESULTS6 or (a.seal_dir / f"gate_verdict_{a.name}.json").exists():
            out += [a.results / r for r in a.adv_names] + [a.cache / r for r in a.adv_names]
    return out


def refuse_sealed_write(path: Path) -> None:
    """Raise SealedWrite if `path` lies in (or is) a protected adversarial folder: any adv_* folder directly under
    results/session_006 or .cache/session_006, or a sealed session_008 attempt's run folders."""
    p = Path(path).resolve()
    for top in (RESULTS6, CACHE6):
        top = top.resolve()
        if p.is_relative_to(top) and p != top and p.relative_to(top).parts[0].startswith("adv_"):
            raise SealedWrite(f"{p}: nothing writes under the session_006 adversarial folders, which A1's committed seal hashes")
    for f in protected_folders():
        if p.is_relative_to(f.resolve()):
            raise SealedWrite(f"{p}: nothing writes under an adversarial run's folders after its attempt's gate verdict")
