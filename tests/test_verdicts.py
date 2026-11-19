"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text). Die Netze sind die Mini-Instanzen aus conftest.py (von Hand gerechnet); die übrigen Eingaben sind Zeilen,
keine Zufalls-Seeds. Dazu: jedes Preset und die Ränder durchlaufen, ein Preset ohne Effekt meldet keinen Effekt."""

from dataclasses import replace

import numpy as np
import fsp_constants as C
import fsp_core as K
import fsp_evaluation as ev
import fsp_net as N
import fsp_scenario as S


def _a(net):
    return ev.analyse_net(net)


# ------------------------------------------------------------------------------------------------------------------------- Kernfrage
def test_three_edges_outside_with_surplus_edge_named(drei):
    kind, text = ev.verdict_shapley_core(_a(drei))
    assert kind == "warning" and "**außerhalb**" in text
    assert "Kante B hat Grenzbeitrag 0" in text                    # B ist überzählig
    assert "im Shapley-Wert zusammen 0,500 von 4" in text           # 1/2 von v(N) = 4
    assert "Koalition {A, C} bekäme 3,500 statt 4" in text          # 5/2 + 1 = 7/2 < v(AC) = 4


def test_parallel_net_is_a_success_with_zero_margin(parallel):
    kind, text = ev.verdict_shapley_core(_a(parallel))
    assert kind == "success" and "**im Kern**" in text and "Überschuss 0)" in text


def test_series_net_is_a_success_because_every_edge_is_a_veto_edge(reihe):
    assert ev.verdict_shapley_core(_a(reihe))[0] == "success"


def test_four_edges_is_the_rest_case_without_surplus_edge(vier):
    a = _a(vier)
    assert a.surplus == () and a.convex is False
    kind, text = ev.verdict_shapley_core(a)
    assert kind == "warning" and "Restfall" in text and "nicht konvex" in text
    assert "Koalition {A, D} bekäme 4,667 statt 5" in text          # 17/6 + 11/6 = 14/3 < 5


def test_bridge_net_with_unit_capacity_has_the_bridge_as_surplus_edge(bruecke):
    a = _a(bruecke)
    assert a.surplus == (2,)
    kind, text = ev.verdict_shapley_core(a)
    assert kind == "warning" and "Kante C hat Grenzbeitrag 0" in text and "zusammen 0,033 von 2" in text      # 1/30


def test_two_surplus_edges_use_the_plural():
    net = N.Net(2, ((0, 1, 3), (0, 1, 3), (0, 1, 3)))                # drei gleiche parallele Kanten: keine überzählig
    assert _a(net).surplus == ()
    net = N.Net(3, ((0, 1, 1), (1, 2, 5), (1, 2, 5), (1, 2, 5)))     # B, C, D parallel mit je 5, A mit 1 davor: alle drei überzählig
    a = _a(net)
    assert a.surplus == (1, 2, 3)
    assert "Kante B, C, D haben Grenzbeitrag 0" in ev.verdict_shapley_core(a)[1]


def test_no_flow_is_a_warning_without_numbers_of_a_split():
    kind, text = ev.verdict_shapley_core(_a(N.Net(2, ((0, 1, 0),))))
    assert kind == "warning" and "v(N) = 0" in text


def test_single_edge_net_gets_the_whole_value():
    kind, text = ev.verdict_shapley_core(_a(N.Net(2, ((0, 1, 3),))))
    assert kind == "info" and "ganze Wert 3" in text


def test_the_contradiction_branch_is_an_error():
    a = replace(_a(N.Net(3, ((0, 1, 4), (1, 2, 3), (1, 2, 5)), "x")), margin=0.0)       # künstlich: im Kern trotz überzähliger Kante
    kind, text = ev.verdict_shapley_core(a)
    assert kind == "error" and "Widerspruch" in text


def test_in_core_threshold_is_inclusive_at_minus_1e9():
    base = _a(N.Net(2, ((0, 1, 2), (0, 1, 3))))
    assert replace(base, margin=-1e-9).in_core and not replace(base, margin=-1.1e-9).in_core


# ------------------------------------------------------------------------------------------------------------------------ Hülle, Spielraum
def test_hull_verdict_equal_and_gap():
    kind, text = ev.verdict_hull({"directions": 50, "with_gap": 0, "equal": True, "max_gap_rel": 0.0, "n_cuts": 3})
    assert kind == "success" and "In allen 50 getesteten Richtungen" in text and "3 minimalen Schnitte" in text
    kind, text = ev.verdict_hull({"directions": 46, "with_gap": 23, "equal": False, "max_gap_rel": 0.25, "n_cuts": 1})
    assert kind == "info" and "**größer**" in text and "in 23 von 46 Richtungen" in text and "um 25,0 % von v(N)" in text


def test_ranges_verdict_names_full_and_zero_edges(drei):
    a = _a(drei)
    kind, text = ev.verdict_ranges(a, K.core_ranges(a.v))
    assert kind == "info"
    assert "Kante A kann im Kern den ganzen Wert 4 beanspruchen" in text            # A ist Veto-Kante: Kern-Obergrenze 4 = v(N), Untergrenze 3
    assert "Kante B bekommt im Kern höchstens 0" in text


def test_ranges_verdict_full_edges_in_a_series_net(reihe):
    a = _a(reihe)
    text = ev.verdict_ranges(a, K.core_ranges(a.v))[1]
    assert "Kante A, B, C kann im Kern den ganzen Wert 2 beanspruchen" in text


def test_ranges_verdict_without_extremes_and_without_flow():
    a = _a(S.example_net("Brücke (3, 2, 1, 2, 3)"))
    ranges = np.array([[0.0, 3.0]] * 5)
    kind, text = ev.verdict_ranges(a, ranges)
    assert "Keine Kante kann den ganzen Wert beanspruchen" in text
    assert ev.verdict_ranges(_a(N.Net(2, ((0, 1, 0),))), np.zeros((1, 2)))[0] == "warning"


# ----------------------------------------------------------------------------------------------------------------------------- Ausbau
def _ausbau(shapley, core_min):
    return {"caps": [1, 2, 3], "shapley": shapley, "core_min": core_min, "core_max": [4, 4, 4],
            "shapley_falls": bool(np.diff(shapley).min() < 0), "core_min_falls": bool(np.diff(core_min).min() < 0)}


def test_ausbau_verdict_opposite_directions():
    kind, text = ev.verdict_ausbau(_ausbau([0.5, 1.0, 2.5], [1.0, 1.0, 0.0]), "A")
    assert kind == "warning" and "Gegenläufig" in text and "von 1 (Kapazität 2) auf 0 (Kapazität 3)" in text


def test_ausbau_verdict_both_rise_and_shapley_falls():
    assert ev.verdict_ausbau(_ausbau([0.5, 1.0, 2.5], [0.0, 1.0, 1.0]), "B")[0] == "info"
    kind, text = ev.verdict_ausbau(_ausbau([2.5, 1.0, 0.5], [0.0, 1.0, 1.0]), "B")
    assert kind == "error" and "widerspricht" in text


def test_ausbau_verdict_on_the_real_example_of_the_plan():
    """Kante s -> 4 (Kapazität c) vor 4 -> t = (4, 1) parallel, daneben s -> t = (3, 5): Kern-Untergrenze 1 für c = 1 bis 4, 0 ab c = 5 (aus der Vormessung, hier neu gerechnet)."""
    import fsp_experiments as E

    net = N.Net(3, ((0, 1, 1), (1, 2, 4), (1, 2, 1), (0, 2, 3), (0, 2, 5)), "x")
    res = E.monotonicity_edge(net, 0, caps=range(1, 9))
    assert res["shapley_falls"] is False and res["core_min_falls"] is True and res["core_max_falls"] is False
    np.testing.assert_allclose(res["core_min"], [1, 1, 1, 1, 0, 0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(res["shapley"], [2 / 3, 7 / 6, 5 / 3, 13 / 6, 2.5, 2.5, 2.5, 2.5], atol=1e-9)
    kind, text = ev.verdict_ausbau(res, "A")
    assert kind == "warning" and "von 1 (Kapazität 4) auf 0 (Kapazität 5)" in text


# ----------------------------------------------------------------------------------------------------------------------- Messreihen (E1 bis E4)
def _cells(rates):
    return [{"k": k, "in_core_mean": r, "runs": 100} for k, r in rates]


def test_e1_verdict_names_first_and_last_and_the_first_zero():
    kind, text = ev.verdict_e1(_cells([(3, 0.59), (4, 0.31), (10, 0.0), (12, 0.0)]))
    assert kind == "success" and "Bei k = 3 Kanten liegt der Shapley-Wert in 59,0 % der Netze im Kern, bei k = 12 in 0,0 %" in text
    assert "Ab k = 10 gab es in dieser Messreihe keinen Treffer." in text and "(100 bzw. 100 Netze)" in text


def test_e1_verdict_no_zero_and_rising_series_is_only_info():
    kind, text = ev.verdict_e1(_cells([(3, 0.2), (4, 0.4)]))
    assert kind == "info" and "keinen Treffer" not in text


def test_e1_zero_tail_must_hold_for_every_larger_k():
    text = ev.verdict_e1(_cells([(3, 0.5), (4, 0.0), (5, 0.1), (6, 0.0)]))[1]
    assert "Ab k = 6 gab es" in text


def test_e2_verdict_compares_unit_with_unequal_capacities():
    rows = [{"capmode": "alle 1", "equal_mean": 1.0, "runs": 100}, {"capmode": "1 bis 5", "equal_mean": 0.24, "runs": 100}, {"capmode": "1 bis 20", "equal_mean": 0.19, "runs": 100}]
    kind, text = ev.verdict_e2(rows)
    assert kind == "info" and "Bei Kapazität 1 stimmen Kern und Hülle in 100,0 % der 100 Netze überein" in text
    assert "1 bis 5: 24,0 %, 1 bis 20: 19,0 %" in text


def test_e3_verdict_names_each_count():
    text = ev.verdict_e3({"pairs": 304, "ks": [5, 6, 8], "shapley_falls": 0, "core_min_falls": 50, "core_max_falls": 24})[1]
    assert "In 304 Kante-Netz-Paaren (k = 5, 6, 8)" in text and "0-mal, die Kern-Untergrenze 50-mal, die Kern-Obergrenze 24-mal" in text


def test_rowgen_verdict_statuses():
    ok = K.RowGenResult(np.array([0.0, 3.0, 1.0]), rows=14, rounds=5, status="optimal")
    kind, text = ev.verdict_rowgen(ok, 3, 1)
    assert kind == "success" and "Kante B im Kern: 3" in text and "14 Zeilen statt 6" in text and "5 Runden" in text
    big = K.RowGenResult(np.zeros(40), rows=60, rounds=30, status="optimal")
    assert "statt 1.099.511.627.774" in ev.verdict_rowgen(big, 40, 0)[1]
    kind, text = ev.verdict_rowgen(K.RowGenResult(np.zeros(3), 9, 4, "Zeitgrenze der Trennung"), 3, 0)
    assert kind == "warning" and "**nicht** als optimal bewiesen" in text and "nach 4 Runden und 9 Zeilen" in text
    kind, text = ev.verdict_rowgen(K.RowGenResult(np.zeros(3), 9, 4, "Löserfehler der Trennung"), 3, 0)
    assert kind == "error" and "Löserfehler der Trennung nach 4 Runden und 9 Zeilen" in text and "**nicht** als optimal bewiesen" in text and not text.startswith("Zeitgrenze")
    kind, text = ev.verdict_rowgen(K.RowGenResult(None, 9, 4, "unzulässig"), 3, 0)
    assert kind == "error" and "Status: unzulässig" in text


# -------------------------------------------------------------------------------------------------------- Formatierung und Presets
def test_number_formats_use_the_decimal_comma():
    assert ev.fmt(4.0) == "4" and ev.fmt(0.5) == "0,500" and ev.fmt(-1 / 3) == "-0,333" and ev.fmt(1 - 1e-12) == "1"
    assert ev.pct(0.1234) == "12,3 %" and ev.pct(0) == "0,0 %"


def test_every_preset_gets_the_expected_kind_and_a_text_that_matches_the_net():
    expected = {"Drei Kanten (Standard)": "warning", "Reihe (5, 2, 3)": "success", "Parallel (2, 3)": "success", "Brücke mit Einheitskapazität": "warning",
                "Brücke (3, 2, 1, 2, 3)": "success", "Vier Kanten (5, 1 | 3, 5)": "warning", "Zufallsnetz mit 8 Kanten": "warning",
                "Zufallsnetz ohne überzählige Kante": "success"}
    assert set(expected) == set(C.PRESETS)
    for name, preset in C.PRESETS.items():
        a = ev.analyse(ev.Settings(**preset).effective())
        kind, text = ev.verdict_shapley_core(a)
        assert kind == expected[name], (name, text)
        assert ("außerhalb" in text) == (kind == "warning")


def test_a_preset_without_effect_reports_no_effect():
    """Parallel (2, 3) und die Reihe: kein Konflikt, also keine Warnung und keine verletzte Koalition im Text."""
    for name in ("Parallel (2, 3)", "Reihe (5, 2, 3)"):
        text = ev.verdict_shapley_core(ev.analyse(ev.Settings(**C.PRESETS[name]).effective()))[1]
        assert "bekäme" not in text and "Grenzbeitrag 0" not in text


def test_verdicts_run_at_the_slider_edges():
    for form in C.RANDOM_FORMS:
        for k in (C.K_MIN, C.K_MAX):
            for capmode in C.CAPMODES:
                a = ev.analyse(ev.Settings(form=form, k=k, capmode=capmode, seed=0))
                kind, text = ev.verdict_shapley_core(a)
                assert kind in ("success", "warning") and text
