"""Einheiten des Kerns an den Mini-Instanzen von Hand (DEMO-PLAYBOOK §1b): Fluss, Schnitte, Koalitionswerte, Shapley-Wert, Kern, Zeilenerzeugung.

Jede Einheit ist eine Funktion mit eigenen Ein- und Ausgaben; die Erwartungswerte stehen als Rechnung im Test (v(AC) = min(4, 5) = 4 usw.)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import fsp_constants as C
import fsp_core as K
import fsp_game as G
import fsp_net as N
import fsp_scenario as S

M = {"A": 1, "B": 2, "C": 4}   # Bit der Kanten A, B, C in der Maske


# ----------------------------------------------------------------------------------------------------------------------- Fluss
def test_max_flow_series_is_the_bottleneck(reihe):
    value, flows = N.max_flow(reihe)
    assert value == 2 and flows == [2, 2, 2]


def test_max_flow_parallel_adds(parallel):
    value, flows = N.max_flow(parallel)
    assert value == 5 and flows == [2, 3]


def test_max_flow_of_a_coalition_uses_only_its_edges(drei):
    assert N.max_flow_value(drei, M["A"] | M["B"]) == 3          # min(4, 3)
    assert N.max_flow_value(drei, M["A"] | M["C"]) == 4          # min(4, 5)
    assert N.max_flow_value(drei, M["B"] | M["C"]) == 0          # ohne A kommt nichts aus der Quelle
    assert N.max_flow_value(drei, 0) == 0


def test_max_flow_on_the_bridge_net_pushes_one_unit_per_path(bruecke):
    value, flows = N.max_flow(bruecke)
    assert value == 2
    assert flows[0] <= 1 and flows[3] <= 1 and sum(flows[:2]) == 2 and sum(flows[3:]) == 2
    assert flows[2] == 0          # die Brücke wird nicht gebraucht: zwei Wege ohne sie


def test_max_flow_reports_flow_conservation(vier):
    value, flows = N.max_flow(vier)
    assert flows[0] + flows[1] == value == flows[2] + flows[3]
    assert all(0 <= f <= c for f, c in zip(flows, vier.caps))


# --------------------------------------------------------------------------------------------------------------------- Schnitte
def test_cut_matrix_of_the_series_net_lists_every_source_side_choice(reihe):
    """Innere Knoten 1, 2: Quellenseite {s} -> Kante A; {s,1} -> B; {s,1,2} -> C; {s,2} -> A und C (ein gültiger, aber nicht minimaler Schnitt)."""
    rows = {tuple(int(x) for x in r) for r in N.cut_matrix(reihe)}
    assert rows == {(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1)}


def test_cut_vectors_carry_the_capacities(reihe):
    w = N.cut_vectors(reihe)
    assert sorted(map(tuple, w.tolist())) == [(0.0, 0.0, 3.0), (0.0, 2.0, 0.0), (5.0, 0.0, 0.0), (5.0, 0.0, 3.0)]
    assert sorted(map(tuple, N.cut_vectors(reihe, minimal_only=True).tolist())) == [(0.0, 0.0, 3.0), (0.0, 2.0, 0.0), (5.0, 0.0, 0.0)]


def test_minimal_cut_vectors_of_the_bridge_net(bruecke):
    # drei minimale Schnitte der Kapazität 2: {A, B}, {D, E}, {A, C, E}
    w = {tuple(int(x) for x in r) for r in N.minimal_cut_vectors(bruecke)}
    assert w == {(1, 1, 0, 0, 0), (0, 0, 0, 1, 1), (1, 0, 0, 0, 1)}


def test_dominated_cuts_are_dropped_without_changing_the_values(vier):
    full = N.cut_vectors(vier)
    reduced = N.cut_vectors(vier, minimal_only=True)
    assert len(reduced) <= len(full)
    assert sorted(map(tuple, reduced.tolist())) == [(0.0, 0.0, 3.0, 5.0), (5.0, 1.0, 0.0, 0.0)]     # Quellenseite {s} und {s, 1}
    v_full = N.additive_array(full[0])
    assert len(full) == 2 and v_full.shape == (16,)


def test_too_many_inner_nodes_are_rejected():
    big = N.Net(N.MAX_INNER_NODES_FOR_CUTS + 3, ((0, 1, 1),))
    with pytest.raises(ValueError):
        N.cut_matrix(big)


# --------------------------------------------------------------------------------------------------------------- Koalitionswerte
def test_coalition_values_of_three_edges(drei):
    v = N.coalition_values(drei)
    expected = {0: 0, M["A"]: 0, M["B"]: 0, M["C"]: 0, M["A"] | M["B"]: 3, M["A"] | M["C"]: 4, M["B"] | M["C"]: 0, 7: 4}
    assert {s: v[s] for s in range(8)} == pytest.approx(expected)


def test_coalition_values_of_the_series_net_only_the_grand_coalition_counts(reihe):
    v = N.coalition_values(reihe)
    assert v[:7].tolist() == [0.0] * 7 and v[7] == 2.0


def test_coalition_values_of_the_parallel_net_are_additive(parallel):
    assert N.coalition_values(parallel).tolist() == [0.0, 2.0, 3.0, 5.0]


def test_coalition_values_of_the_four_edge_net(vier):
    v = N.coalition_values(vier)
    a, b, c, d = 1, 2, 4, 8
    assert v[a | d] == 5 and v[a | c] == 3 and v[a | b | c] == 3 and v[a | b | d] == 5 and v[a | b | c | d] == 6   # min(5, 5); min(5, 3); min(6, 3); min(6, 5); min(6, 8)
    assert v[a | b] == 0 and v[c | d] == 0


def test_additive_array_orders_bits_by_edge():
    assert N.additive_array([1.0, 10.0]).tolist() == [0.0, 1.0, 10.0, 11.0]


def test_edge_names_continue_after_z():
    names = N.edge_names(28)
    assert names[:3] == ["A", "B", "C"] and names[25] == "Z" and names[26] == "A2" and names[27] == "B2"
    assert len(set(names)) == 28


def test_coalition_label():
    assert N.coalition_label(0b101, 3) == "{A, C}" and N.coalition_label(0, 3) == "{}"


def test_relevant_edges_skips_dead_ends():
    net = N.Net(4, ((0, 1, 1), (1, 3, 1), (0, 2, 1)))        # Kante 2 führt in eine Sackgasse (Knoten 2 erreicht die Senke nicht)
    assert N.relevant_edges(net) == [0, 1]


def test_check_net_rejects_cycles_loops_and_bad_capacities():
    with pytest.raises(ValueError):
        N.check_net(N.Net(3, ((0, 1, 1), (1, 2, 1), (2, 1, 1))))          # Kreis 1 -> 2 -> 1
    with pytest.raises(ValueError):
        N.check_net(N.Net(3, ((0, 0, 1),)))
    with pytest.raises(ValueError):
        N.check_net(N.Net(3, ((0, 1, -1),)))
    with pytest.raises(ValueError):
        N.check_net(N.Net(1, ()))
    N.check_net(N.Net(3, ((0, 1, 1), (1, 2, 0))))                          # Kapazität 0 ist erlaubt (Nullspieler)


# --------------------------------------------------------------------------------------------------------------- Spielgrößen
@pytest.mark.parametrize("name", ["reihe", "parallel", "drei", "bruecke", "vier"])
def test_shapley_matches_the_hand_values(request, name):
    from conftest import HAND_SHAPLEY, HAND_VALUE_GRAND

    net = request.getfixturevalue(name)
    v = N.coalition_values(net)
    assert v[-1] == HAND_VALUE_GRAND[name]
    assert G.shapley_fractions(v) == HAND_SHAPLEY[name]
    assert G.shapley(v) == pytest.approx([float(x) for x in HAND_SHAPLEY[name]], rel=1e-12)
    assert sum(G.shapley(v)) == pytest.approx(v[-1])               # Effizienz


def test_null_players_are_the_edges_off_every_path():
    net = N.Net(4, ((0, 1, 2), (1, 3, 2), (0, 2, 5)))              # Kante 2 führt nach Knoten 2 und nirgends weiter
    v = N.coalition_values(net)
    assert G.null_players(v) == [False, False, True]
    assert G.shapley(v)[2] == 0.0


def test_surplus_and_veto_edges_of_three_edges(drei):
    v = N.coalition_values(drei)
    assert G.surplus_edges(v) == [1]            # B: v(ABC) - v(AC) = 4 - 4 = 0, kein Nullspieler (v(AB) = 3 > v(A) = 0)
    assert G.veto_edges(v) == [0]               # ohne A ist v = 0
    assert G.grand_marginals(v).tolist() == [4.0, 0.0, 1.0]


def test_series_net_has_only_veto_edges(reihe):
    v = N.coalition_values(reihe)
    assert G.veto_edges(v) == [0, 1, 2] and G.surplus_edges(v) == []


def test_veto_edges_are_empty_when_nothing_flows():
    assert G.veto_edges(np.zeros(4)) == []


def test_convexity_of_parallel_and_series_and_three_edges(parallel, reihe, drei):
    assert G.is_convex(N.coalition_values(parallel))
    assert G.is_convex(N.coalition_values(reihe))
    assert not G.is_convex(N.coalition_values(drei))     # Grenzbeitrag von B: 3 zu {A}, aber 0 zu {A, C}: er wächst nicht mit der Koalition


def test_core_margin_names_the_violated_coalition(vier):
    v = N.coalition_values(vier)
    margin, gap, mask = G.core_margin(v, G.shapley(v))
    assert margin == pytest.approx(-1 / 3) and gap == pytest.approx(0, abs=1e-12)
    assert mask == 1 | 8                                 # {A, D}: x = 17/6 + 11/6 = 14/3 < v = 5


def test_core_margin_single_player():
    assert G.core_margin(np.array([0.0, 7.0]), [7.0]) == (0.0, 0.0, 0)
    assert G.in_core(np.array([0.0, 7.0]), [7.0]) and not G.in_core(np.array([0.0, 7.0]), [6.0])


def test_num_players_requires_a_power_of_two():
    with pytest.raises(ValueError):
        G.num_players(np.zeros(6))


def test_exact_shapley_needs_integer_values_and_few_players():
    with pytest.raises(ValueError):
        G.shapley_fractions(np.array([0.0, 0.5]))
    with pytest.raises(ValueError):
        G.shapley_fractions(np.zeros(1 << (G.EXACT_SHAPLEY_MAX_PLAYERS + 1)))


# ------------------------------------------------------------------------------------------------------------------------- Kern
def test_core_of_three_edges_by_hand(drei):
    v = N.coalition_values(drei)
    ranges = K.core_ranges(v)
    # Kern = {(x_A, 0, 4 - x_A) : x_A >= 3}: A zwischen 3 und 4, B genau 0, C zwischen 0 und 1
    assert_allclose(ranges, [[3, 4], [0, 0], [0, 1]], atol=K.TOL)


def test_core_of_the_series_net_is_the_whole_simplex(reihe):
    ranges = K.core_ranges(N.coalition_values(reihe))
    assert_allclose(ranges, [[0, 2]] * 3, atol=K.TOL)      # H10: jede Veto-Kante kann den ganzen Wert beanspruchen


def test_core_of_the_parallel_net_is_a_single_point(parallel):
    assert_allclose(K.core_ranges(N.coalition_values(parallel)), [[2, 2], [3, 3]], atol=K.TOL)


def test_core_of_the_bridge_net_gives_the_bridge_nothing(bruecke):
    ranges = K.core_ranges(N.coalition_values(bruecke))
    assert ranges[2].tolist() == pytest.approx([0, 0], abs=K.TOL)
    assert ranges[0].tolist() == pytest.approx([0, 1], abs=K.TOL)


def test_core_lp_without_objective_is_a_feasibility_test(drei):
    ok, x = K.core_lp(N.coalition_values(drei))
    assert ok and G.in_core(N.coalition_values(drei), x, tol=K.TOL)


def test_core_lp_refuses_too_many_players():
    with pytest.raises(ValueError):
        K.core_lp(np.zeros(1 << (K.CORE_LP_MAX_PLAYERS + 1)))


def test_core_lp_single_player():
    assert K.core_lp(np.array([0.0, 3.0]))[1].tolist() == [3.0]


def test_min_cut_payoffs_are_in_the_core(bruecke, drei, vier):
    for net in (bruecke, drei, vier):
        v = N.coalition_values(net)
        for x in K.min_cut_payoffs(net, v[-1]):
            assert G.in_core(v, x), (net.name, x)


def test_hull_equals_core_for_unit_capacities(bruecke):
    v = N.coalition_values(bruecke)
    dirs = np.vstack([np.eye(5), -np.eye(5), np.array([[1, -1, 2, -2, 0.5], [0.3, 0.9, -1, 1, 1]])])
    gap = K.hull_gap(bruecke, v, dirs)
    assert gap["equal"] and gap["with_gap"] == 0 and gap["n_cuts"] == 3


def test_hull_is_smaller_than_the_core_for_three_edges(drei):
    """Kern = {(x_A, 0, 4 - x_A) : 3 <= x_A <= 4}, Hülle der minimalen Schnitte = der Punkt (4, 0, 0): in Richtung -e_A (kleinstes x_A) klafft die Lücke 1."""
    v = N.coalition_values(drei)
    gap = K.hull_gap(drei, v, np.array([[-1.0, 0.0, 0.0]]))
    assert not gap["equal"] and gap["with_gap"] == 1 and gap["max_gap_rel"] == pytest.approx(1 / 4)


# --------------------------------------------------------------------------------------------------------------- Zeilenerzeugung
def test_row_generation_by_enumeration_matches_the_full_program(drei, vier):
    for net in (drei, vier):
        v = N.coalition_values(net)
        for i in range(net.m):
            lo, hi = K.core_range(v, i)
            res = K.max_payment_by_row_generation(net, i, "enum", v=v)
            assert res.status == "optimal" and res.x[i] == pytest.approx(hi, abs=K.TOL)


def test_row_generation_by_milp_matches_the_enumeration(drei, bruecke):
    for net in (drei, bruecke):
        v = N.coalition_values(net)
        for i in range(net.m):
            e = K.max_payment_by_row_generation(net, i, "enum", v=v)
            m = K.max_payment_by_row_generation(net, i, "milp")
            assert m.status == "optimal" and m.x[i] == pytest.approx(e.x[i], abs=1e-6), (net.name, i)


def test_separation_by_enumeration_finds_the_most_violated_coalition(vier):
    v = N.coalition_values(vier)
    found = K.separate_enumeration(v)(G.shapley(v))
    assert found is not None and found[0] == 1 | 8 and found[1] == pytest.approx(1 / 3)
    assert K.separate_enumeration(v)(np.array([5.0, 1.0, 0.0, 0.0])) is None          # ein minimaler Schnitt liegt im Kern


def test_separation_returns_none_inside_the_core(drei):
    v = N.coalition_values(drei)
    assert K.separate_enumeration(v)(np.array([4.0, 0.0, 0.0])) is None


def test_separation_by_milp_agrees_with_enumeration_on_the_violation(vier):
    v = N.coalition_values(vier)
    x = G.shapley(v)
    by_enum = K.separate_enumeration(v)(x)
    by_milp = K.separate_milp(vier)(x)
    assert by_milp is not None and by_milp[1] == pytest.approx(by_enum[1], abs=1e-6)
    assert K.separate_milp(vier)(np.array([5.0, 1.0, 0.0, 0.0])) is None


def test_row_generation_reports_a_time_limit_as_status(drei):
    def boom(x):
        raise TimeoutError("Zeitgrenze der Trennung")

    v = N.coalition_values(drei)
    res = K.row_generation(3, v[-1], v, boom, np.array([1.0, 0, 0]))
    assert res.status == "Zeitgrenze der Trennung"


class _FakeMilp:
    """Ersatz für `scipy.optimize.milp`: liefert einen festen Status (scipy: 0 optimal, 1 Zeitgrenze, 2 unzulässig, 3 unbeschränkt, 4 sonstiger Fehler)."""

    def __init__(self, status, with_x=True):
        self.status, self.with_x = status, with_x

    def __call__(self, cost, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(status=self.status, x=np.zeros(len(cost)) if self.with_x else None, message=f"Status {self.status}")


@pytest.mark.parametrize("status, with_x", [(1, True), (1, False)])
def test_separation_by_milp_reports_status_1_as_time_limit(drei, monkeypatch, status, with_x):
    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "milp", _FakeMilp(status, with_x))
    with pytest.raises(TimeoutError, match="Zeitgrenze"):
        K.separate_milp(drei)(np.array([4.0, 0.0, 0.0]))


@pytest.mark.parametrize("status, with_x", [(2, False), (3, False), (4, False), (4, True), (0, False)])
def test_separation_by_milp_reports_other_statuses_as_solver_error_not_as_time_limit(drei, monkeypatch, status, with_x):
    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "milp", _FakeMilp(status, with_x))
    with pytest.raises(K.SeparationSolverError, match=f"Löserstatus {status}"):
        K.separate_milp(drei)(np.array([4.0, 0.0, 0.0]))
    assert not issubclass(K.SeparationSolverError, TimeoutError)


@pytest.mark.parametrize("status, expected", [(1, "Zeitgrenze der Trennung"), (4, "Löserfehler der Trennung"), (2, "Löserfehler der Trennung")])
def test_row_generation_by_milp_labels_the_solver_status(monkeypatch, status, expected):
    """Kette bis zum Status: ein Löserfehler (Status 2 bis 4) erscheint nicht mehr als „Zeitgrenze“."""
    import scipy.optimize

    net = S.example_net(C.EXAMPLES[0])
    monkeypatch.setattr(scipy.optimize, "milp", _FakeMilp(status))
    res = K.max_payment_by_row_generation(net, 0, "milp")
    assert res.status == expected


def test_row_generation_reports_a_solver_error_as_status(drei):
    def boom(x):
        raise K.SeparationSolverError("Löserstatus 4: Fehler")

    v = N.coalition_values(drei)
    res = K.row_generation(3, v[-1], v, boom, np.array([1.0, 0, 0]))
    assert res.status == "Löserfehler der Trennung"


def test_row_generation_reports_infeasible_programs():
    v = np.array([0.0, 5.0, 5.0, 4.0])             # zwei Spieler, die allein 5 bekommen müssten, gemeinsam aber nur 4 haben: Kern leer
    res = K.row_generation(2, 4.0, v, K.separate_enumeration(v), np.array([1.0, 0.0]))
    assert res.status == "unzulässig" and res.x is None


# ------------------------------------------------------------------------------------------------------- Worst-Case am Regler-Maximum
def test_worst_case_at_the_slider_maximum_completes():
    """Kein Zeitvergleich, nur ein Netz gegen Hänger: k = 20 in allen drei Netzformen, alle Koalitionen, exakter Shapley-Wert, Randprüfung."""
    for form in C.RANDOM_FORMS:
        net = S.make_net(form, C.K_MAX, "1 bis 20", 7)
        v = N.coalition_values(net)
        phi = G.shapley(v)
        assert len(v) == 1 << net.m and phi.sum() == pytest.approx(v[-1])
        margin, gap, _ = G.core_margin(v, phi)
        assert gap < 1e-9
