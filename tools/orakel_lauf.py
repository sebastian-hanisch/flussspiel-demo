"""Nachrechenbarer Orakel-Lauf für die README (Abschnitt „Messung dieser README“): feste Netzmenge, feste Zufalls-Seeds, Zähler und Abweichungen.

Aufruf aus dem Demo-Ordner: python tools/orakel_lauf.py   (rund eine Minute; braucht networkx, siehe requirements-dev.txt)

Netzmenge: die sechs Handbeispiele und alle Netze mit den drei Netzformen, den drei Kapazitätsmodi, den Kantenzahlen k = 2 bis 8 und den
Zufalls-Seeds 1 bis 6 (`make_net(Form, k, Kapazitätsmodus, Zufalls-Seed)`).

Zwei Prüfungen:
  1. Koalitionswert (Schnittformel) gegen networkx, scipy und den eigenen Max-Flow auf JEDER Koalition jedes Netzes.
  2. Shapley-Wert (Gewichtsformel, Brüche) gegen die Aufzählung aller Reihenfolgen in Brüchen, für Netze mit höchstens 6 Kanten (720 Reihenfolgen)."""

import itertools
import math
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp_sparse  # noqa: E402
import scipy.sparse.csgraph as sp_graph  # noqa: E402

import fsp_constants as C  # noqa: E402
import fsp_game as G  # noqa: E402
import fsp_net as N  # noqa: E402
import fsp_scenario as S  # noqa: E402

KS = tuple(range(2, 9))
SEEDS = tuple(range(1, 7))
SHAPLEY_MAX_M = 6


def nx_value(net, mask) -> float:
    h = nx.DiGraph()
    h.add_nodes_from(range(net.n))
    for i, (a, b, c) in enumerate(net.edges):
        if mask >> i & 1:
            if h.has_edge(a, b):
                h[a][b]["capacity"] += c
            else:
                h.add_edge(a, b, capacity=c)
    return float(nx.maximum_flow_value(h, 0, net.n - 1)) if h.number_of_edges() else 0.0


def scipy_value(net, mask) -> float:
    mat = np.zeros((net.n, net.n), dtype=np.int32)
    for i, (a, b, c) in enumerate(net.edges):
        if mask >> i & 1:
            mat[a, b] += int(c)
    return float(sp_graph.maximum_flow(sp_sparse.csr_matrix(mat), 0, net.n - 1).flow_value)


def shapley_by_orders(v) -> list:
    m = G.num_players(v)
    phi = [Fraction(0)] * m
    for perm in itertools.permutations(range(m)):
        coalition = 0
        for i in perm:
            phi[i] += Fraction(int(v[coalition | 1 << i])) - Fraction(int(v[coalition]))
            coalition |= 1 << i
    return [p / math.factorial(m) for p in phi]


def net_set() -> list:
    nets = [S.example_net(name) for name in C.EXAMPLES]
    for form in C.RANDOM_FORMS:
        for capmode in C.CAPMODES:
            for k in KS:
                for seed in SEEDS:
                    nets.append(S.make_net(form, k, capmode, seed))
    return nets


def run(nets=None) -> dict:
    """Zähler: Netze, Koalitionen, Abweichungen der Schnittformel, Netze und Abweichungen des Shapley-Vergleichs."""
    nets = net_set() if nets is None else nets
    out = {"nets": len(nets), "coalitions": 0, "cut_deviations": 0, "shapley_nets": 0, "shapley_deviations": 0}
    for net in nets:
        v = N.coalition_values(net)
        for mask in range(1 << net.m):
            out["coalitions"] += 1
            if not (v[mask] == nx_value(net, mask) == scipy_value(net, mask) == N.max_flow_value(net, mask)):
                out["cut_deviations"] += 1
        if net.m <= SHAPLEY_MAX_M:
            out["shapley_nets"] += 1
            out["shapley_deviations"] += G.shapley_fractions(v) != shapley_by_orders(v)
    return out


def main() -> None:
    res = run()
    print(f"Netze: {res['nets']}, Koalitionen: {res['coalitions']}, Abweichungen der Schnittformel: {res['cut_deviations']}")
    print(f"Shapley-Wert gegen alle Reihenfolgen: {res['shapley_nets']} Netze (höchstens {SHAPLEY_MAX_M} Kanten), Abweichungen: {res['shapley_deviations']}")


if __name__ == "__main__":
    main()
