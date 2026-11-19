"""Kern des Flussspiels: Linearprogramm über alle Koalitionen, Spielraum je Kante, Schnitt-Auszahlungen, Hülle, Zeilenerzeugung.

scipy wird erst in den Funktionen importiert, die es brauchen (kürzerer Kaltstart). HiGHS liefert keine exakten Gleitkommawerte: Zahlen aus den
Programmen gelten nur bis etwa 1e-7; Tests und Verdikte arbeiten mit Toleranz (TOL) und Bändern."""

import time
from dataclasses import dataclass

import numpy as np

from fsp_game import core_margin, num_players
from fsp_net import Net, additive_array, max_flow_value, minimal_cut_vectors

TOL = 1e-7                  # Toleranz für Werte aus HiGHS und für „Kern-Maximum gleich 0“
CORE_LP_MAX_PLAYERS = 14    # volles Programm mit 2^m - 2 Zeilen (dicht) bis hierher
SEPARATION_ENUM_MAX = 20    # Trennung durch Aufzählung aller 2^m Koalitionen bis hierher


def _proper_masks(m: int) -> np.ndarray:
    return np.arange(1, (1 << m) - 1)


def core_lp(v, c=None, sense: str = "min") -> tuple:
    """Zielfunktion c·x über dem Kern, alle Zeilen. Rückgabe (zulässig, x). Ohne c: reine Zulässigkeitsfrage. Höchstens CORE_LP_MAX_PLAYERS Spieler."""
    from scipy.optimize import linprog

    v = np.asarray(v, dtype=float)
    m = num_players(v)
    if m > CORE_LP_MAX_PLAYERS:
        raise ValueError(f"das volle Programm rechnet nur bis {CORE_LP_MAX_PLAYERS} Spieler (danach Zeilenerzeugung)")
    if m == 1:
        return True, np.array([v[-1]])
    masks = _proper_masks(m)
    a_ub = -((masks[:, None] >> np.arange(m)[None, :]) & 1).astype(float)
    b_ub = -v[masks]
    cost = np.zeros(m) if c is None else (np.asarray(c, float) if sense == "min" else -np.asarray(c, float))
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=[np.ones(m)], b_eq=[v[-1]], bounds=[(None, None)] * m, method="highs")
    return res.status == 0, (res.x if res.status == 0 else None)


def core_nonempty(v) -> bool:
    return core_lp(v)[0]


def core_range(v, i: int) -> tuple:
    """(kleinste, größte) Zahlung an Spieler i im Kern; Zeilenerzeugung ab CORE_LP_MAX_PLAYERS + 1 Spielern (Trennung durch Aufzählung)."""
    v = np.asarray(v, dtype=float)
    m = num_players(v)
    c = np.zeros(m)
    c[i] = 1.0
    if m <= CORE_LP_MAX_PLAYERS:
        _, lo = core_lp(v, c, "min")
        _, hi = core_lp(v, c, "max")
        return float(lo[i]), float(hi[i])
    lo = row_generation(m, v[-1], v, separate_enumeration(v), -c)
    hi = row_generation(m, v[-1], v, separate_enumeration(v), c)
    return float(lo.x[i]), float(hi.x[i])


def core_ranges(v) -> np.ndarray:
    """Spielraum je Spieler: Zeile i = (Kern-Minimum, Kern-Maximum)."""
    m = num_players(v)
    return np.array([core_range(v, i) for i in range(m)])


def min_cut_payoffs(net: Net, v_grand: float | None = None) -> np.ndarray:
    """Zeilen = Auszahlungsvektoren der minimalen Schnitte (x_e = c_e auf dem Schnitt, sonst 0); sie liegen im Kern."""
    return minimal_cut_vectors(net, v_grand)


def directions(m: int, count: int, rng) -> np.ndarray:
    """`count` Zufallsrichtungen (Einträge gleichverteilt auf [-1, 1], SplitMix64) plus die 2m Koordinatenrichtungen ±e_i."""
    rand = np.array([[2.0 * rng.uniform() - 1.0 for _ in range(m)] for _ in range(count)]).reshape(count, m)
    return np.vstack([rand, np.eye(m), -np.eye(m)])


def hull_gap(net: Net, v, dirs) -> dict:
    """Kern gegen die konvexe Hülle der minimalen Schnitt-Auszahlungen, getestet in den Richtungen `dirs`.

    In jeder Richtung w wird das Maximum von w·x über dem Kern (Programm) mit dem besten Schnittvektor verglichen. Die Hülle liegt immer im Kern;
    eine Lücke in irgendeiner Richtung zeigt, dass der Kern größer ist. Gleich in allen getesteten Richtungen heißt nur: in diesen Richtungen kein Unterschied."""
    v = np.asarray(v, dtype=float)
    cuts = min_cut_payoffs(net, v[-1])
    gaps = []
    for w in dirs:
        best_cut = float(np.max(cuts @ w))
        _, x = core_lp(v, w, "max")
        gaps.append(float(w @ x) - best_cut)
    gaps = np.array(gaps)
    scale = max(float(v[-1]), 1e-12)
    return {
        "directions": len(gaps),
        "with_gap": int(np.sum(gaps > TOL)),
        "equal": bool(np.all(gaps <= TOL)),
        "max_gap_rel": float(gaps.max() / scale) if len(gaps) else 0.0,
        "n_cuts": int(len(cuts)),
    }


# ------------------------------------------------------------------------------------------------------------------ Zeilenerzeugung
class SeparationSolverError(RuntimeError):
    """Das gemischt-ganzzahlige Trennungsproblem endete mit einem anderen Status als optimal (0) oder Zeitgrenze (1)."""


@dataclass
class RowGenResult:
    x: np.ndarray | None
    rows: int           # Zahl der Koalitionszeilen im Programm am Ende
    rounds: int
    status: str         # "optimal" | "unzulässig" | "Rundengrenze" | "Zeitgrenze der Trennung" | "Löserfehler der Trennung"
    separation_seconds: float = 0.0
    total_seconds: float = 0.0


def separate_enumeration(v):
    """Trennungsproblem durch Aufzählung: die am stärksten verletzte Koalition (x(S) - v(S) minimal) oder None."""
    v = np.asarray(v, dtype=float)

    def separate(x, tol=TOL):
        d = additive_array(x) - v
        d[0] = 0.0
        d[-1] = 0.0
        j = int(np.argmin(d))
        return (j, float(-d[j])) if d[j] < -tol else None

    return separate


def separate_milp(net: Net, time_limit: float = 20.0):
    """Trennungsproblem als gemischt-ganzzahliges Fixkostenflussproblem: max_S v(S) - x(S).

    Variablen: Fluss f_e in [0, c_e] und y_e in {0, 1} (Kante gehört zu S); f_e <= c_e y_e; Flusserhaltung an inneren Knoten;
    Zielwert Nettofluss aus der Quelle minus Σ x_e y_e. Gibt (Koalition, Verletzung) oder None; die Verletzung wird mit dem eigenen Max-Flow
    nachgerechnet (nicht aus dem Programm übernommen). Meldet das Programm die Zeitgrenze (Status 1), wird `TimeoutError` geworfen, bei jedem
    anderen Status außer 0 (optimal) `SeparationSolverError`."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    m = net.m
    n = net.n
    caps = np.array(net.caps, dtype=float)
    a_eq = np.zeros((max(n - 2, 0), 2 * m))
    for i, (a, b, _) in enumerate(net.edges):
        if 1 <= a <= n - 2:
            a_eq[a - 1, i] -= 1
        if 1 <= b <= n - 2:
            a_eq[b - 1, i] += 1
    a_ub = np.zeros((m, 2 * m))
    for i in range(m):
        a_ub[i, i] = 1
        a_ub[i, m + i] = -caps[i]
    constraints = [LinearConstraint(a_ub, -np.inf, 0)]
    if n > 2:
        constraints.append(LinearConstraint(a_eq, 0, 0))
    integrality = np.concatenate([np.zeros(m), np.ones(m)])
    bounds = Bounds(np.zeros(2 * m), np.concatenate([caps, np.ones(m)]))
    out_of_source = np.zeros(2 * m)
    for i, (a, b, _) in enumerate(net.edges):
        if a == 0:
            out_of_source[i] += 1
        if b == 0:
            out_of_source[i] -= 1

    def separate(x, tol=TOL):
        x = np.maximum(np.asarray(x, dtype=float), 0.0)   # x >= 0 sichern die Einzelzeilen (v({i}) >= 0) bis auf Rundung
        cost = np.concatenate([-out_of_source[:m], x])
        res = milp(cost, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": time_limit, "mip_rel_gap": 0})
        if res.status == 1:                          # scipy: 1 = Iterations- oder Zeitgrenze erreicht (mit oder ohne Näherungslösung)
            raise TimeoutError("Zeitgrenze der Trennung")
        if res.status != 0 or res.x is None:         # 2 unzulässig, 3 unbeschränkt, 4 sonstiger Löserfehler: keine Zeitgrenze
            raise SeparationSolverError(f"Löserstatus {res.status}: {res.message}")
        mask = sum(1 << i for i in range(m) if res.x[m + i] > 0.5)
        violation = max_flow_value(net, mask) - float(sum(x[i] for i in range(m) if mask >> i & 1))
        return (mask, float(violation)) if violation > tol else None

    return separate


def row_generation(m: int, v_grand: float, value_of, separate, c, max_rounds: int = 2000) -> RowGenResult:
    """Maximiere c·x über dem Kern per Zeilenerzeugung.

    Start: die Einzelspieler-Zeilen x_i >= v({i}); je Runde wird das Programm gelöst und `separate(x)` sucht die am stärksten verletzte
    Koalition; keine verletzte Koalition heißt, x liegt im Kern und ist optimal. `value_of(S)` liefert v(S) (Array-Index oder Funktion)."""
    from scipy.optimize import linprog

    t_start = time.perf_counter()
    get = value_of.__getitem__ if hasattr(value_of, "__getitem__") else value_of
    full = (1 << m) - 1
    rows = [1 << i for i in range(m) if (1 << i) != full]
    cost = -np.asarray(c, dtype=float)
    sep_seconds = 0.0
    for rnd in range(1, max_rounds + 1):
        masks = np.array(rows, dtype=np.int64)
        a_ub = -((masks[:, None] >> np.arange(m)[None, :]) & 1).astype(float) if len(rows) else np.zeros((0, m))
        b_ub = -np.array([float(get(int(s))) for s in rows]) if len(rows) else np.zeros(0)
        res = linprog(cost, A_ub=a_ub if len(rows) else None, b_ub=b_ub if len(rows) else None, A_eq=[np.ones(m)], b_eq=[v_grand],
                      bounds=[(None, None)] * m, method="highs")
        if res.status != 0:
            return RowGenResult(None, len(rows), rnd, "unzulässig", sep_seconds, time.perf_counter() - t_start)
        t0 = time.perf_counter()
        try:
            found = separate(res.x)
        except TimeoutError:
            sep_seconds += time.perf_counter() - t0
            return RowGenResult(res.x, len(rows), rnd, "Zeitgrenze der Trennung", sep_seconds, time.perf_counter() - t_start)
        except SeparationSolverError:
            sep_seconds += time.perf_counter() - t0
            return RowGenResult(res.x, len(rows), rnd, "Löserfehler der Trennung", sep_seconds, time.perf_counter() - t_start)
        sep_seconds += time.perf_counter() - t0
        if found is None:
            return RowGenResult(res.x, len(rows), rnd, "optimal", sep_seconds, time.perf_counter() - t_start)
        rows.append(int(found[0]))
    return RowGenResult(None, len(rows), max_rounds, "Rundengrenze", sep_seconds, time.perf_counter() - t_start)


def max_payment_by_row_generation(net: Net, i: int, method: str = "auto", v=None, time_limit: float = 20.0) -> RowGenResult:
    """Größte Zahlung an Kante i im Kern. `method`: "enum" (Aufzählung, bis SEPARATION_ENUM_MAX Kanten), "milp" oder "auto" (Aufzählung wenn möglich)."""
    m = net.m
    if method == "auto":
        method = "enum" if m <= SEPARATION_ENUM_MAX else "milp"
    c = np.zeros(m)
    c[i] = 1.0
    if method == "enum":
        if v is None:
            from fsp_net import coalition_values

            v = coalition_values(net)
        return row_generation(m, v[-1], v, separate_enumeration(v), c)
    if method == "milp":
        grand = float(max_flow_value(net))
        return row_generation(m, grand, lambda s: max_flow_value(net, s), separate_milp(net, time_limit), c)
    raise ValueError(method)


def check_core_point(net: Net, x, v) -> bool:
    """Prüfer für die Anzeige: liegt x im Kern (alle Koalitionen, Toleranz TOL)?"""
    margin, gap, _ = core_margin(v, x)
    return margin >= -TOL and gap <= TOL
