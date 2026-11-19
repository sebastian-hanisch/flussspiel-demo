"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): ein, zwei Spieler, Gleichstände, Nullspieler, Kapazität 0, Reglerränder."""

import numpy as np
import pytest

import fsp_constants as C
import fsp_core as K
import fsp_evaluation as ev
import fsp_game as G
import fsp_net as N
import fsp_scenario as S


def test_single_edge_net():
    net = N.Net(2, ((0, 1, 3),))
    v = N.coalition_values(net)
    assert v.tolist() == [0.0, 3.0] and G.shapley(v).tolist() == [3.0]
    assert G.core_margin(v, G.shapley(v)) == (0.0, 0.0, 0)
    assert K.core_range(v, 0) == (3.0, 3.0) and K.core_ranges(v).tolist() == [[3.0, 3.0]]
    a = ev.analyse_net(net)
    assert a.in_core and a.veto == (0,) and a.surplus == ()


def test_exactly_two_edges_in_series_and_in_parallel():
    """n = 2: der kleinste Fall mit etwas zu vergleichen (Zweige `m < 2` und die leere Menge echter Koalitionen dürfen nicht falsch greifen)."""
    series = N.coalition_values(N.Net(3, ((0, 1, 2), (1, 2, 5))))
    assert series.tolist() == [0.0, 0.0, 0.0, 2.0] and G.shapley(series).tolist() == [1.0, 1.0]
    np.testing.assert_allclose(K.core_ranges(series), [[0, 2], [0, 2]], atol=K.TOL)
    parallel = N.coalition_values(N.Net(2, ((0, 1, 2), (0, 1, 3))))
    assert G.shapley(parallel).tolist() == [2.0, 3.0] and G.in_core(parallel, G.shapley(parallel))


def test_equal_parallel_edges_are_symmetric_ties():
    """Gleichstand: drei gleiche parallele Kanten bekommen gleich viel (Shapley-Wert und Kern fallen auf einen Punkt)."""
    net = N.Net(2, ((0, 1, 2),) * 3)
    v = N.coalition_values(net)
    assert G.shapley(v).tolist() == pytest.approx([2.0, 2.0, 2.0])
    np.testing.assert_allclose(K.core_ranges(v), [[2, 2]] * 3, atol=K.TOL)


def test_two_identical_serial_bottlenecks_share_the_value_equally():
    v = N.coalition_values(N.Net(3, ((0, 1, 4), (1, 2, 4))))
    assert G.shapley(v).tolist() == [2.0, 2.0] and G.veto_edges(v) == [0, 1]


def test_zero_capacity_edge_is_a_null_player_and_gets_nothing_anywhere():
    net = N.Net(2, ((0, 1, 4), (0, 1, 0)))
    v = N.coalition_values(net)
    assert G.null_players(v) == [False, True]
    assert G.shapley(v).tolist() == [4.0, 0.0] and K.core_range(v, 1) == pytest.approx((0.0, 0.0), abs=K.TOL)
    assert G.surplus_edges(v) == []                    # ein Nullspieler ist nicht überzählig


def test_a_net_without_a_path_has_value_zero_and_only_null_players():
    net = N.Net(4, ((0, 1, 5), (2, 3, 5)))
    v = N.coalition_values(net)
    assert v.max() == 0.0 and all(G.null_players(v)) and G.veto_edges(v) == [] and G.surplus_edges(v) == []
    assert N.max_flow(net) == (0, [0, 0])
    assert ev.verdict_shapley_core(ev.analyse_net(net))[0] == "warning"


def test_the_empty_coalition_and_the_grand_coalition_are_excluded_from_the_margin():
    v = N.coalition_values(N.Net(2, ((0, 1, 2), (0, 1, 3))))
    # x = (2, 3) füllt jede echte Koalition genau aus: Überschuss 0, nicht -5 (leere Koalition) und nicht 0 durch die große Koalition verdeckt
    assert G.core_margin(v, [2.0, 3.0]) == (0.0, 0.0, 1)
    margin, gap, _ = G.core_margin(v, [2.0, 4.0])
    assert margin == 0.0 and gap == 1.0                 # nicht effizient: Abstand 1


def test_row_generation_with_two_players_and_with_one():
    v = np.array([0.0, 1.0, 2.0, 5.0])
    res = K.row_generation(2, 5.0, v, K.separate_enumeration(v), np.array([1.0, 0.0]))
    assert res.status == "optimal" and res.x[0] == pytest.approx(3.0)
    one = K.row_generation(1, 3.0, np.array([0.0, 3.0]), K.separate_enumeration(np.array([0.0, 3.0])), np.array([1.0]))
    assert one.status == "optimal" and one.x[0] == pytest.approx(3.0) and one.rows == 0


def test_tolerance_is_exact_at_the_edge_with_fractions():
    """Der Shapley-Wert eines Spiels mit Kern-Rand-Lösung liegt mit Überschuss genau 0 im Kern (Parallel), mit -1/3 knapp außerhalb (Vier Kanten)."""
    parallel = N.coalition_values(N.Net(2, ((0, 1, 2), (0, 1, 3))))
    assert G.in_core(parallel, G.shapley(parallel))
    vier = N.coalition_values(S.example_net("Vier Kanten (5, 1 | 3, 5)"))
    margin, _, _ = G.core_margin(vier, G.shapley(vier))
    assert margin == pytest.approx(-1 / 3) and not G.in_core(vier, G.shapley(vier))
    assert G.in_core(vier, G.shapley(vier), tol=0.34)


@pytest.mark.parametrize("form", C.FORMS)
def test_every_net_form_with_two_edges(form):
    k = 2
    net = S.make_net(form, k, "1 bis 5", 3)
    if form in C.RANDOM_FORMS:
        assert net.m == 2
    a = ev.analyse_net(net)
    assert a.v_grand >= 0 and np.isfinite(a.phi).all()
    assert ev.verdict_shapley_core(a)[0] in ("success", "warning", "info")


@pytest.mark.parametrize("form", C.RANDOM_FORMS)
@pytest.mark.parametrize("capmode", list(C.CAPMODES))
def test_slider_corners_all_forms_and_modes(form, capmode):
    for k in (C.K_MIN, C.K_MAX):
        net = S.make_net(form, k, capmode, C.SEED_MAX)
        assert net.m == k
    net = S.make_net(form, C.K_MIN, capmode, 0)
    v = N.coalition_values(net)
    assert G.shapley(v).sum() == pytest.approx(v[-1])


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        G.shapley_fractions(np.array([0.0, 1.0, 2.5, 3.0]))
    with pytest.raises(ValueError):
        N.check_net(N.Net(2, ((0, 5, 1),)))
    with pytest.raises(ValueError):
        N.check_net(N.Net(2, ((0, 1, 1.5),)))
    with pytest.raises(ValueError):
        K.max_payment_by_row_generation(S.example_net("Reihe (5, 2, 3)"), 0, "unbekannt")
