"""Orakel-Tests (DEMO-PLAYBOOK §12): jede Kernroutine gegen einen anderen Rechenweg.

  * Koalitionswerte (Schnittformel) gegen networkx und scipy auf JEDER Koalition kleiner Netze und gegen den eigenen Max-Flow auf Stichproben großer;
  * Shapley-Wert (Gewichtsformel, vektorisiert) gegen die Aufzählung aller Reihenfolgen in Brüchen;
  * Konvexität (Paarbedingung) gegen die Definition v(S ∪ T) + v(S ∩ T) >= v(S) + v(T) über alle Paare von Koalitionen;
  * Nullspieler, überzählige und Veto-Kanten gegen ihre Definitionen in Brute Force;
  * minimale Schnitte gegen Aufzählung aller Quellenseiten in reinem Python;
  * Kern: Lösungen des Programms gegen die Prüfung aller Koalitionen, Zeilenerzeugung (Aufzählung und gemischt-ganzzahlige Trennung) gegen das volle Programm;
  * Max-Flow-Min-Cut-Gleichheit und Schnitt-Auszahlungen im Kern.
Die größere, nachrechenbare Fassung der Koalitionswert- und Shapley-Prüfung steht in `tools/orakel_lauf.py`; hier die kleine, schnelle (feste Zufalls-Seeds, unter 10 s)."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

import fsp_constants as C
import fsp_core as K
import fsp_game as G
import fsp_net as N
import fsp_scenario as S

nx = pytest.importorskip("networkx")
sp_graph = pytest.importorskip("scipy.sparse.csgraph")
sp_sparse = pytest.importorskip("scipy.sparse")


def _nx_value(net, mask):
    h = nx.DiGraph()
    h.add_nodes_from(range(net.n))
    for i, (a, b, c) in enumerate(net.edges):
        if mask >> i & 1:
            if h.has_edge(a, b):
                h[a][b]["capacity"] += c
            else:
                h.add_edge(a, b, capacity=c)
    return float(nx.maximum_flow_value(h, 0, net.n - 1)) if h.number_of_edges() else 0.0


def _scipy_value(net, mask):
    mat = np.zeros((net.n, net.n), dtype=np.int32)
    for i, (a, b, c) in enumerate(net.edges):
        if mask >> i & 1:
            mat[a, b] += int(c)
    return float(sp_graph.maximum_flow(sp_sparse.csr_matrix(mat), 0, net.n - 1).flow_value)


def _small_nets(count=36, max_k=6):
    nets = []
    for j in range(count):
        form = C.RANDOM_FORMS[j % 3]
        capmode = list(C.CAPMODES)[(j // 3) % 3]
        k = 2 + (j % (max_k - 1))
        nets.append(S.make_net(form, k, capmode, 100 + j))
    return nets


SMALL = _small_nets()


def test_the_small_nets_cover_all_forms_and_capacity_modes():
    assert {n.name for n in SMALL} == {"Zufalls-DAG", "Reihen-Parallel-Netz", "Engpasskette"}
    assert {n.m for n in SMALL} == {2, 3, 4, 5, 6}


def test_cut_formula_equals_networkx_and_scipy_on_every_coalition():
    checked = 0
    for net in SMALL:
        v = N.coalition_values(net)
        for mask in range(1 << net.m):
            assert v[mask] == _nx_value(net, mask) == _scipy_value(net, mask) == N.max_flow_value(net, mask), (net, mask)
            checked += 1
    assert checked == sum(1 << n.m for n in SMALL)


def test_hand_examples_equal_networkx_on_every_coalition():
    for name in C.EXAMPLES:
        net = S.example_net(name)
        v = N.coalition_values(net)
        assert [v[s] for s in range(1 << net.m)] == [_nx_value(net, s) for s in range(1 << net.m)], name


def test_cut_formula_equals_the_own_max_flow_on_samples_of_large_nets():
    for form in C.RANDOM_FORMS:
        net = S.make_net(form, 14, "1 bis 5", 5)
        v = N.coalition_values(net)
        for mask in np.random.default_rng(3).integers(0, 1 << net.m, size=60):
            assert v[mask] == N.max_flow_value(net, int(mask)) == _scipy_value(net, int(mask))


def test_max_flow_min_cut_value_of_the_grand_coalition_equals_networkx_minimum_cut():
    for net in SMALL[:20]:
        h = nx.DiGraph()
        for a, b, c in net.edges:
            if h.has_edge(a, b):
                h[a][b]["capacity"] += c
            else:
                h.add_edge(a, b, capacity=c)
        cut_value, _ = nx.minimum_cut(h, 0, net.n - 1)
        assert N.coalition_values(net)[-1] == cut_value


# -------------------------------------------------------------------------------------------------------------------------- Shapley
def _shapley_by_permutations(v):
    m = G.num_players(v)
    phi = [Fraction(0)] * m
    for perm in itertools.permutations(range(m)):
        coalition = 0
        for i in perm:
            phi[i] += Fraction(int(v[coalition | 1 << i])) - Fraction(int(v[coalition]))
            coalition |= 1 << i
    return [p / math.factorial(m) for p in phi]


def test_shapley_formula_equals_the_enumeration_of_all_orders_in_fractions():
    for net in [n for n in SMALL if n.m <= 6]:
        v = N.coalition_values(net)
        by_perm = _shapley_by_permutations(v)
        assert G.shapley_fractions(v) == by_perm, net
        np.testing.assert_allclose(G.shapley(v), [float(x) for x in by_perm], rtol=1e-12, atol=1e-12)


def test_hand_examples_shapley_equals_the_enumeration():
    for name in C.EXAMPLES:
        v = N.coalition_values(S.example_net(name))
        assert G.shapley_fractions(v) == _shapley_by_permutations(v), name


# ------------------------------------------------------------------------------------------------- Konvexität, Null, überzählig, Veto
def _is_convex_by_definition(v):
    m = G.num_players(v)
    return all(v[s | t] + v[s & t] >= v[s] + v[t] - 1e-12 for s in range(1 << m) for t in range(s, 1 << m))


def test_convexity_equals_the_definition_over_all_pairs_of_coalitions():
    convex = nonconvex = 0
    for net in [n for n in SMALL if n.m <= 5]:
        v = N.coalition_values(net)
        want = _is_convex_by_definition(v)
        assert G.is_convex(v) == want, net
        convex += want
        nonconvex += not want
    assert convex > 0 and nonconvex > 0


def test_null_surplus_and_veto_edges_equal_their_definitions():
    for net in SMALL:
        v = N.coalition_values(net)
        m, full = net.m, net.full
        null = [all(v[s | 1 << i] == v[s] for s in range(1 << m) if not s >> i & 1) for i in range(m)]
        surplus = [i for i in range(m) if not null[i] and v[full] == v[full ^ (1 << i)]]
        veto = [i for i in range(m) if v[full] > 0 and v[full ^ (1 << i)] == 0]
        assert G.null_players(v) == null and G.surplus_edges(v) == surplus and G.veto_edges(v) == veto, net


# ------------------------------------------------------------------------------------------------------------------ minimale Schnitte
def _min_cut_payoffs_by_enumeration(net):
    inner = list(range(1, net.n - 1))
    best, payoffs = None, set()
    for r in range(len(inner) + 1):
        for chosen in itertools.combinations(inner, r):
            side = {0, *chosen}
            w = tuple(c if (a in side and b not in side) else 0 for a, b, c in net.edges)
            total = sum(w)
            if best is None or total < best:
                best, payoffs = total, {w}
            elif total == best:
                payoffs.add(w)
    return best, payoffs


def test_minimal_cut_payoffs_equal_the_enumeration_of_all_source_sides():
    for net in SMALL:
        best, payoffs = _min_cut_payoffs_by_enumeration(net)
        got = {tuple(int(x) for x in row) for row in K.min_cut_payoffs(net)}
        assert N.coalition_values(net)[-1] == best and got == payoffs, net


def test_minimal_cut_payoffs_lie_in_the_core_by_the_check_of_all_coalitions():
    total = 0
    for net in SMALL:
        v = N.coalition_values(net)
        for x in K.min_cut_payoffs(net):
            assert G.in_core(v, x), (net, x)
            total += 1
    assert total >= len(SMALL)


# ------------------------------------------------------------------------------------------------------------------------------ Kern
def test_core_program_returns_points_that_pass_the_check_of_all_coalitions():
    for net in [n for n in SMALL if n.m <= 6]:
        v = N.coalition_values(net)
        ok, x = K.core_lp(v)
        assert ok and G.in_core(v, x, tol=K.TOL), net
        for i in range(net.m):
            lo, hi = K.core_range(v, i)
            assert -K.TOL <= lo <= hi + K.TOL <= v[-1] + 2 * K.TOL


def test_core_is_never_empty_in_the_whole_and_in_every_subgame():
    """Totale Ausgewogenheit: auch jede Koalition als eigenes Spiel hat einen nichtleeren Kern (Netze mit höchstens 4 Kanten, alle 2^m - 1 Teilspiele)."""
    subgames = 0
    for net in [n for n in SMALL if n.m <= 4]:
        v = N.coalition_values(net)
        for s in range(1, 1 << net.m):
            idx = [i for i in range(net.m) if s >> i & 1]
            sub = np.array([v[sum(1 << idx[b] for b in range(len(idx)) if t >> b & 1)] for t in range(1 << len(idx))])
            assert K.core_nonempty(sub), (net, s)
            subgames += 1
    assert subgames > 100


def test_surplus_edges_get_zero_in_the_core_and_a_positive_shapley_value():
    seen = 0
    for net in SMALL:
        v = N.coalition_values(net)
        phi = G.shapley(v)
        for i in G.surplus_edges(v):
            assert K.core_range(v, i)[1] == pytest.approx(0, abs=K.TOL) and phi[i] > 1e-9
            assert not G.in_core(v, phi), net
            seen += 1
    assert seen >= 3


def test_veto_edges_can_claim_the_whole_value_in_the_core():
    seen = 0
    for net in SMALL:
        v = N.coalition_values(net)
        for i in G.veto_edges(v):
            x = np.zeros(net.m)
            x[i] = v[-1]
            assert G.in_core(v, x), (net, i)
            assert K.core_range(v, i)[1] == pytest.approx(v[-1], abs=K.TOL)
            seen += 1
    assert seen >= 3


def test_row_generation_equals_the_full_program_on_larger_nets():
    for k in (9, 11):
        net = S.make_net("Zufalls-DAG", k, "1 bis 5", 40 + k)
        v = N.coalition_values(net)
        for i in range(0, k, 3):
            lo, hi = K.core_range(v, i)
            res = K.max_payment_by_row_generation(net, i, "enum", v=v)
            assert res.status == "optimal" and res.x[i] == pytest.approx(hi, abs=K.TOL)
            assert res.rows < (1 << k) - 2


def test_milp_separation_equals_the_enumeration_for_a_batch_of_points():
    for net in SMALL[:12]:
        v = N.coalition_values(net)
        enum, milp = K.separate_enumeration(v), K.separate_milp(net)
        for x in (G.shapley(v), np.full(net.m, v[-1] / net.m)):
            a, b = enum(x), milp(x)
            assert (a is None) == (b is None), (net, x)
            if a is not None:
                assert b[1] == pytest.approx(a[1], abs=1e-6)
