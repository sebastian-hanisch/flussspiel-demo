"""Vorgerechnete Messreihen: Konfiguration, Aufbau und Lesen von `data/messreihen.json`.

Die Datei enthält den Hash der Quelltexte, aus denen sie entstand (`source_hash`); `tests/test_measurements.py` bricht, wenn Kern, Netzerzeuger,
Zufallsquelle oder Messroutinen seither geändert wurden: dann `python tools/make_messreihen.py` neu laufen lassen und die Zahlen in README und
`tests/test_claims.py` nachziehen. Auf einer anderen Plattform entstehen dieselben Netze (SplitMix64, Ganzzahlen); die Anteile hängen nur an
Vergleichen mit Toleranz 1e-9. Laufzeiten stehen in der Datei als Information, nie in einem Test."""

import hashlib
import json
import time
from pathlib import Path

import fsp_constants as C
import fsp_experiments as E

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "messreihen.json"
SOURCES = ("fsp_net.py", "fsp_game.py", "fsp_core.py", "fsp_scenario.py", "fsp_rng.py", "fsp_experiments.py", "fsp_constants.py", "fsp_tables.py")

E1_DAG_KS = (3, 4, 5, 6, 8, 10, 12, 14)
E1_DAG_BIG = (16, 18, 20)
E1_OTHER_KS = (3, 4, 5, 6, 8, 10, 12)
E1_RUNS, E1_BIG_RUNS = 25, 3
E2_KS, E2_RUNS = (4, 5, 6, 7, 8), 5
CLASS_KS, CLASS_RUNS = (4, 5, 6, 7, 8), 4
RESTFALL_KS = (3, 4, 5, 6, 7, 8)
ROWGEN = ((10, "enum", 5, True), (12, "enum", 5, True), (14, "enum", 4, True), (20, "enum", 3, False), (30, "milp", 6, False), (40, "milp", 6, False))


def source_hash() -> str:
    """SHA-256 über die Quelltexte (Zeilenenden vereinheitlicht, damit Windows und Linux denselben Wert bekommen)."""
    h = hashlib.sha256()
    for name in SOURCES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def build(progress=None) -> dict:
    """Alle Messreihen neu rechnen (rund 2 Minuten)."""
    say = progress or (lambda *_: None)
    t0 = time.perf_counter()
    out: dict = {"e1": [], "e2": {"hull": [], "classes": []}, "e3": None, "restfall": None, "e4": []}
    for form in C.RANDOM_FORMS:
        for capmode in ("alle 1", "1 bis 5"):
            ks = (E1_DAG_KS + E1_DAG_BIG) if form == "Zufalls-DAG" else E1_OTHER_KS
            for k in ks:
                runs = E1_BIG_RUNS if k in E1_DAG_BIG else E1_RUNS
                out["e1"].append(E.shapley_core_cell(form, capmode, k, runs))
            say("E1", form, capmode)
    for capmode in C.CAPMODES:
        out["e2"]["hull"].append(E.hull_cell(capmode, E2_KS, E2_RUNS))
        say("E2 Hülle", capmode)
    for capmode in ("alle 1", "1 bis 5"):
        out["e2"]["classes"].append(E.range_classes(capmode, CLASS_KS, CLASS_RUNS))
        say("E2 Klassen", capmode)
    out["e3"] = E.monotonicity_table()
    say("E3")
    out["restfall"] = E.restfall_table(RESTFALL_KS)
    say("Restfall")
    for m, method, runs, compare in ROWGEN:
        out["e4"].append(E.rowgen_cell(m, runs, method, compare_full=compare and m <= 14))
        say("E4", m)
    out["meta"] = {"source_hash": source_hash(), "build_seconds": round(time.perf_counter() - t0, 1), "sweep_seeds": list(C.SWEEP_SEEDS),
                   "e1_runs": E1_RUNS, "e1_big_runs": E1_BIG_RUNS}
    return out


def load() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def is_current(data: dict | None = None) -> bool:
    data = load() if data is None else data
    return data.get("meta", {}).get("source_hash") == source_hash()


def e1_cell(data: dict, form: str, capmode: str, k: int) -> dict | None:
    return next((c for c in data["e1"] if c["form"] == form and c["capmode"] == capmode and c["k"] == k), None)
