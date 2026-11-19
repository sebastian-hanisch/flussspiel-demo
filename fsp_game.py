"""Kooperatives Spiel auf den Koalitionswerten: Shapley-Wert, Nullspieler, überzählige und Veto-Kanten, Konvexität, Kernprüfung.

Das Spiel ist ein **Gewinnspiel** (Erlös je durchgesetzter Einheit, kein Kostenspiel): die Kernbedingung heißt x(S) >= v(S) für jede Koalition
S und x(N) = v(N). Die Prüfung über alle Koalitionen schließt die große Koalition aus (dort gilt die Gleichung) und die leere.
`v` ist ein Array der Länge 2^m, Bit i der Koalitionsmaske ist Spieler i."""

import math
from fractions import Fraction

import numpy as np

from fsp_net import additive_array

EXACT_SHAPLEY_MAX_PLAYERS = 12   # exakte Brüche: Zähler bleiben in int64


def num_players(v) -> int:
    m = len(v).bit_length() - 1
    if (1 << m) != len(v):
        raise ValueError("die Länge von v muss eine Zweierpotenz sein")
    return m


def popcount(m: int) -> np.ndarray:
    """Zahl der Spieler jeder Koalitionsmaske 0 ... 2^m - 1."""
    pc = np.zeros(1, dtype=np.int64)
    for _ in range(m):
        pc = np.concatenate([pc, pc + 1])
    return pc


def shapley(v) -> np.ndarray:
    """Shapley-Wert: Mittel des Grenzbeitrags über alle Reihenfolgen, per Koalitionsgewicht s!(m-s-1)!/m! (vektorisiert über alle Koalitionen)."""
    v = np.asarray(v, dtype=float)
    m = num_players(v)
    if m == 0:
        return np.zeros(0)
    pc = popcount(m)
    f = [math.factorial(i) for i in range(m + 1)]
    weight = np.array([f[s] * f[m - s - 1] / f[m] if s <= m - 1 else 0.0 for s in range(m + 1)])
    idx = np.arange(1 << m)
    phi = np.zeros(m)
    for i in range(m):
        without = idx[(idx >> i & 1) == 0]
        phi[i] = np.sum(weight[pc[without]] * (v[without | (1 << i)] - v[without]))
    return phi


def shapley_fractions(v) -> list:
    """Exakter Shapley-Wert als Brüche (nur ganzzahlige v, höchstens EXACT_SHAPLEY_MAX_PLAYERS Spieler)."""
    m = num_players(v)
    if m > EXACT_SHAPLEY_MAX_PLAYERS:
        raise ValueError(f"exakte Brüche nur bis {EXACT_SHAPLEY_MAX_PLAYERS} Spieler")
    vi = np.rint(np.asarray(v, dtype=float)).astype(np.int64)
    if not np.array_equal(vi, np.asarray(v, dtype=float)):
        raise ValueError("exakte Brüche brauchen ganzzahlige Koalitionswerte")
    pc = popcount(m)
    idx = np.arange(1 << m)
    f = [math.factorial(i) for i in range(m + 1)]
    weight = np.array([f[s] * f[m - s - 1] if s <= m - 1 else 0 for s in range(m + 1)], dtype=np.int64)
    out = []
    for i in range(m):
        without = idx[(idx >> i & 1) == 0]
        numerator = int(np.sum(weight[pc[without]] * (vi[without | (1 << i)] - vi[without])))
        out.append(Fraction(numerator, f[m]))
    return out


def null_players(v) -> list:
    """Spieler i ist Nullspieler, wenn v(S + i) = v(S) für alle Koalitionen S ohne i."""
    v = np.asarray(v, dtype=float)
    m = num_players(v)
    idx = np.arange(1 << m)
    result = []
    for i in range(m):
        without = idx[(idx >> i & 1) == 0]
        result.append(bool(np.all(np.abs(v[without | (1 << i)] - v[without]) < 1e-12)))
    return result


def grand_marginals(v) -> np.ndarray:
    """Grenzbeitrag jedes Spielers zur großen Koalition: v(N) - v(N ohne i)."""
    v = np.asarray(v, dtype=float)
    m = num_players(v)
    full = (1 << m) - 1
    return np.array([v[full] - v[full ^ (1 << i)] for i in range(m)])


def surplus_edges(v) -> list:
    """Überzählige Spieler: kein Nullspieler, aber ohne sie sinkt der Wert der großen Koalition nicht (Grenzbeitrag 0 zu N)."""
    gb = grand_marginals(v)
    null = null_players(v)
    return [i for i in range(len(gb)) if not null[i] and gb[i] < 1e-12]


def veto_edges(v) -> list:
    """Veto-Spieler: ohne sie ist die große Koalition wertlos (v(N ohne i) = 0 bei v(N) > 0)."""
    v = np.asarray(v, dtype=float)
    m = num_players(v)
    full = (1 << m) - 1
    if v[full] <= 0:
        return []
    return [i for i in range(m) if v[full ^ (1 << i)] < 1e-12]


def is_convex(v) -> bool:
    """Konvex (supermodular): v(S + i + j) - v(S + j) >= v(S + i) - v(S) für alle S und alle Paare i != j außerhalb S."""
    v = np.asarray(v, dtype=float)
    m = num_players(v)
    idx = np.arange(1 << m)
    for i in range(m):
        for j in range(i + 1, m):
            rest = idx[((idx >> i) & 1 == 0) & ((idx >> j) & 1 == 0)]
            lhs = v[rest | (1 << i) | (1 << j)] - v[rest | (1 << j)]
            rhs = v[rest | (1 << i)] - v[rest]
            if np.any(lhs < rhs - 1e-12):
                return False
    return True


coalition_sums = additive_array   # x(S) für alle Koalitionen


def core_margin(v, x) -> tuple:
    """(kleinster Überschuss x(S) - v(S) über alle echten Koalitionen, Abstand |x(N) - v(N)|, Maske der am stärksten verletzten Koalition).

    Bei einem Spieler gibt es keine echte Koalition: der Überschuss ist dann 0. Eine Verteilung liegt im Kern, wenn der kleinste Überschuss >= 0
    (mit Toleranz) und der Effizienzabstand 0 ist."""
    v = np.asarray(v, dtype=float)
    xs = coalition_sums(x)
    if len(v) <= 2:
        return 0.0, float(abs(xs[-1] - v[-1])), 0
    d = xs - v
    inner = d[1:-1]
    j = int(np.argmin(inner)) + 1
    return float(inner.min()), float(abs(xs[-1] - v[-1])), j


def in_core(v, x, tol: float = 1e-9) -> bool:
    margin, gap, _ = core_margin(v, x)
    return margin >= -tol and gap <= tol
