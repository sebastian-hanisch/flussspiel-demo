"""Jede Zahl der README steht hier: die Zahlen der Messreihen kommen aus data/messreihen.json (Hash der Quelltexte, test_measurements.py prüft Aktualität und
rechnet kleine Zellen neu), die Handrechnungen und Zählungen werden hier neu gerechnet. Die README-Sätze werden aus denselben Werten gebildet und müssen wörtlich in
der README stehen. Was nur als „Messung dieser README, nicht im Test“ dort steht (Laufzeiten, Scratch-Läufe), steht auch nur dort."""

import itertools
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

import fsp_constants as C
import fsp_core as K
import fsp_game as G
import fsp_net as N
import fsp_scenario as S
import fsp_tables as T
from fsp_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent
README = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())


@pytest.fixture(scope="module")
def data():
    return T.load()


def pc(x: float) -> str:
    return f"{round(100 * x)} %"


def _cell(data, form, cap, k):
    return T.e1_cell(data, form, cap, k)


# ---------------------------------------------------------------------------------------------------------------------- Handrechnungen
def test_hand_examples_shapley_values_and_core_violations():
    expected = {
        "Reihe (5, 2, 3)": ("2", ["2/3", "2/3", "2/3"]),
        "Parallel (2, 3)": ("5", ["2", "3"]),
        "Drei Kanten (4 | 3, 5)": ("4", ["5/2", "1/2", "1"]),
        "Brücke mit Einheitskapazität": ("2", ["8/15", "9/20", "1/30", "9/20", "8/15"]),
        "Brücke (3, 2, 1, 2, 3)": ("5", ["4/3", "1", "1/3", "1", "4/3"]),
        "Vier Kanten (5, 1 | 3, 5)": ("6", ["17/6", "1/2", "5/6", "11/6"]),
    }
    for name, (value, phi) in expected.items():
        v = N.coalition_values(S.example_net(name))
        assert str(int(v[-1])) == value and [str(f) for f in G.shapley_fractions(v)] == phi, name
        line = f"{name}: v(N) = {value}, Shapley-Wert ({', '.join(phi)})"
        assert line in README, line
    drei = N.coalition_values(S.example_net("Drei Kanten (4 | 3, 5)"))
    assert F(5, 2) + F(1) == F(7, 2) < 4 and drei[0b101] == 4          # x(A, C) = 7/2 < v(A, C) = 4
    vier = N.coalition_values(S.example_net("Vier Kanten (5, 1 | 3, 5)"))
    assert F(17, 6) + F(11, 6) == F(14, 3) and vier[0b1001] == 5 and 5 - F(14, 3) == F(1, 3)
    assert "x(A, D) = 14/3 < v(A, D) = 5" in README and "Verletzung 1/3" in README


def test_hand_example_classes():
    drei = N.coalition_values(S.example_net("Drei Kanten (4 | 3, 5)"))
    assert G.veto_edges(drei) == [0] and G.surplus_edges(drei) == [1]
    brücke = N.coalition_values(S.example_net("Brücke mit Einheitskapazität"))
    assert G.surplus_edges(brücke) == [2] and K.core_range(brücke, 2) == pytest.approx((0, 0), abs=K.TOL)
    vier = N.coalition_values(S.example_net("Vier Kanten (5, 1 | 3, 5)"))
    assert G.surplus_edges(vier) == [] and all(g > 0 for g in G.grand_marginals(vier))


# ----------------------------------------------------------------------------------------------------------------------------- E1
CAP_LABEL = {"alle 1": "Kapazität 1", "1 bis 5": "Kapazitäten 1 bis 5"}


def test_e1_cells_and_sentences(data):
    cells = data["e1"]
    assert len(cells) == 50 and sum(c["runs"] for c in cells) == 4472
    assert all(c["runs"] == (12 if c["k"] >= 16 else 100) for c in cells)
    assert "4 472 Netze in 50 Zellen (je 100 Netze bis k = 14, je 12 für k = 16, 18, 20)" in README
    for form, cap, ks in (("Zufalls-DAG", "alle 1", (3, 4, 5, 6, 8, 10)), ("Zufalls-DAG", "1 bis 5", (3, 4, 5, 6, 8)),
                          ("Reihen-Parallel-Netz", "alle 1", (3, 12)), ("Engpasskette", "alle 1", (3, 10, 12))):
        line = f"{form}, {CAP_LABEL[cap]}: " + ", ".join(f"k = {k}: {pc(_cell(data, form, cap, k)['in_core_mean'])}" for k in ks) + "."
        assert line in README, line


def test_e1_in_the_core_values_by_the_numbers_of_the_readme(data):
    assert pc(_cell(data, "Zufalls-DAG", "alle 1", 3)["in_core_mean"]) == "59 %" and pc(_cell(data, "Zufalls-DAG", "alle 1", 10)["in_core_mean"]) == "0 %"
    assert all(_cell(data, "Zufalls-DAG", cap, k)["in_core_mean"] == 0 for cap in ("alle 1", "1 bis 5") for k in (12, 14, 16, 18, 20))
    assert all(_cell(data, "Zufalls-DAG", "1 bis 5", k)["in_core_mean"] == 0 for k in (8, 10))              # H5: „0 ab k = 8“ bei Kapazitäten 1 bis 5
    assert "Im Zufalls-DAG gab es in dieser Messreihe ab k = 12 keinen Treffer (k = 12, 14: je 100 Netze; k = 16 bis 20: je 12 Netze; beide Kapazitätsmodi)" in README


def test_e1_over_all_six_series_start_high_and_end_low(data):
    for form in C.RANDOM_FORMS:
        for cap in ("alle 1", "1 bis 5"):
            assert round(_cell(data, form, cap, 3)["in_core_mean"], 2) >= 0.59 and round(_cell(data, form, cap, 12)["in_core_mean"], 2) <= 0.09
    at3 = [round(100 * _cell(data, f, c, 3)["in_core_mean"]) for f in C.RANDOM_FORMS for c in ("alle 1", "1 bis 5")]
    assert (min(at3), max(at3)) == (59, 84)
    assert "bei k = 3 mindestens 59 %, bei k = 12 höchstens 9 %" in README
    assert "bei k = 3 zwischen 59 und 84 % und bei k = 12 bei höchstens 9 %" in README


def test_e1_the_series_of_bottleneck_chains_is_not_monotone(data):
    ser = [_cell(data, "Engpasskette", "alle 1", k)["in_core_mean"] for k in (3, 4, 5, 6, 8, 10, 12)]
    assert ser[5] < ser[6]                                                   # 2 % bei k = 10, 9 % bei k = 12
    assert "Engpasskette mit Kapazität 1: 2 % bei k = 10, aber 9 % bei k = 12" in README


def test_e1_surplus_and_in_the_core_never_together(data):
    assert sum(c["surplus_and_in_core"] for c in data["e1"]) == 0
    assert "in keiner der 50 Zellen (4 472 Netze) zugleich" in README


def test_e1_share_of_nets_with_a_surplus_edge(data):
    assert pc(_cell(data, "Zufalls-DAG", "alle 1", 3)["surplus_share"]) == "41 %"
    assert all(_cell(data, "Zufalls-DAG", "alle 1", k)["surplus_share"] == 1.0 for k in (12, 14, 16, 18, 20))
    assert "41 % bei k = 3 und 100 % ab k = 12" in README


def test_the_dag_generator_never_needed_the_fill_in_on_the_checked_nets():
    used = sum(S.random_dag_with_info(SplitMix64(seed), k, "1 bis 5")[1] for k in range(C.K_MIN, C.K_MAX + 1) for seed in range(1, 6))
    assert used == 0
    assert "In 95 geprüften Netzen (alle k von 2 bis 20, 5 Zufalls-Seeds) brauchte der Erzeuger die Auffüllung nie" in README


# ----------------------------------------------------------------------------------------------------------------------------- E2
def test_e2_hull_equals_core_shares(data):
    by = {r["capmode"]: r for r in data["e2"]["hull"]}
    assert [r["runs"] for r in by.values()] == [100, 100, 100] and by["alle 1"]["directions"] == 56
    assert (40 + 2 * min(T.E2_KS), 40 + 2 * max(T.E2_KS)) == (48, 56)                  # je Netz 40 Zufallsrichtungen plus 2k Koordinatenrichtungen, k = 4 bis 8
    assert "(48 bis 56 Richtungen)" in README
    assert (pc(by["alle 1"]["equal_mean"]), pc(by["1 bis 5"]["equal_mean"]), pc(by["1 bis 20"]["equal_mean"])) == ("100 %", "24 %", "19 %")
    assert "Kapazität 1: 100 %, Kapazitäten 1 bis 5: 24 %, Kapazitäten 1 bis 20: 19 %" in README
    assert "je 100 Netze (Zufalls-DAG, k = 4 bis 8)" in README


def test_e2_range_classes(data):
    unit, five = data["e2"]["classes"]
    assert len(C.SWEEP_SEEDS) * len(T.CLASS_KS) * T.CLASS_RUNS == 80 and "je 80 Netze je Kapazitätsmodus" in README
    assert (unit["veto"]["edges"], five["veto"]["edges"]) == (17, 15) and unit["veto"]["max_mean"] == pytest.approx(1.0) and five["veto"]["max_mean"] == pytest.approx(1.0)
    assert (unit["surplus"]["edges"], five["surplus"]["edges"]) == (234, 171)
    assert unit["surplus"]["max_is_zero"] == 234 and five["surplus"]["max_is_zero"] == 171
    assert (f"{unit['other']['max_mean']:.2f}", f"{five['other']['max_mean']:.2f}") == ("0.40", "0.35")
    assert (unit["other"]["edges"], five["other"]["edges"], five["other"]["max_is_zero"]) == (229, 294, 5)
    for needle in ("Veto-Kanten (17 und 15 Kanten)", "überzählige Kanten (234 und 171 Kanten)", "Kern-Obergrenze im Mittel 0,40 und 0,35 (229 und 294 Kanten)",
                   "5 von 294 übrigen Kanten bei Kapazitäten 1 bis 5 haben die Obergrenze 0"):
        assert needle in README, needle


# ----------------------------------------------------------------------------------------------------------------------------- E3
def test_e3_capacity_monotonicity(data):
    e3 = data["e3"]
    assert (e3["pairs"], e3["shapley_falls"], e3["core_min_falls"], e3["core_max_falls"]) == (304, 0, 50, 24) and e3["ks"] == [5, 6, 8]
    assert "304 Kante-Netz-Paaren (k = 5, 6, 8)" in README
    assert "Shapley-Wert fällt in 0, die Kern-Untergrenze in 50, die Kern-Obergrenze in 24 Paaren" in README


def test_e3_the_example_of_the_plan_is_recomputed():
    from fsp_experiments import monotonicity_edge

    net = N.Net(3, ((0, 1, 1), (1, 2, 4), (1, 2, 1), (0, 2, 3), (0, 2, 5)))
    r = monotonicity_edge(net, 0, caps=range(1, 9))
    np.testing.assert_allclose(r["core_min"], [1, 1, 1, 1, 0, 0, 0, 0], atol=1e-7)
    assert r["shapley"][0] == pytest.approx(2 / 3) and r["shapley"][4] == pytest.approx(2.5)
    assert "Kern-Untergrenze 1 für die Kapazitäten 1 bis 4, 0 ab 5, Shapley-Wert 2/3 bis 5/2" in README


# --------------------------------------------------------------------------------------------------------------------------- Restfall
def test_restfall_counts(data):
    rf = data["restfall"]
    assert (rf["nets"], rf["convex"], rf["convex_in_core"], rf["clean"], rf["clean_outside"], rf["clean_outside_nonconvex"], rf["surplus_and_in_core"]) == (600, 141, 141, 185, 22, 22, 0)
    assert "600 Netze: 141 konvexe Spiele, alle 141 mit Shapley-Wert im Kern; 185 Netze ohne überzählige Kante, in 22 davon liegt er außerhalb, alle 22 nicht konvex" in README


# --------------------------------------------------------------------------------------------------------- totale Ausgewogenheit, H10
def test_total_balancedness_over_all_subgames():
    nets = [S.make_net("Zufalls-DAG", k, "1 bis 5", 1 + j) for j in range(16) for k in (4, 5, 6, 7)]
    assert len(nets) == 64
    subgames = empty = 0
    for net in nets:
        v = N.coalition_values(net)
        for s in range(1, 1 << net.m):
            idx = [i for i in range(net.m) if s >> i & 1]
            sub = np.array([v[sum(1 << idx[b] for b in range(len(idx)) if t >> b & 1)] for t in range(1 << len(idx))])
            subgames += 1
            empty += len(idx) > 1 and not K.core_nonempty(sub)
    assert (subgames, empty) == (3776, 0)
    assert "0 leere Kerne in 3 776 Teilspielen (64 Netze, k = 4 bis 7, Zufalls-DAG, Kapazitäten 1 bis 5" in README


# ------------------------------------------------------------------------------------------------------------------------------- E4
def test_e4_rows_of_the_row_generation(data):
    e4 = {r["m"]: r for r in data["e4"]}
    assert (e4[40]["rows_min"], e4[40]["rows_max"], e4[40]["runs"]) == (58, 145, 6) and (e4[30]["rows_min"], e4[30]["rows_max"]) == (42, 79)
    assert (e4[20]["rows_min"], e4[20]["rows_max"]) == (25, 26) and all(r["all_optimal"] for r in e4.values())
    assert sum(e4[m]["equal_to_full"] for m in (10, 12, 14)) == 14 and sum(e4[m]["runs"] for m in (10, 12, 14)) == 14
    for needle in ("m = 40: 58 bis 145 Zeilen", "m = 30: 42 bis 79 Zeilen", "m = 20: 25 bis 26 Zeilen", "gleich dem vollen Programm in 14 von 14 Netzen (m = 10, 12, 14)"):
        assert needle in README, needle


def test_e4_milp_separation_equals_the_enumeration_on_larger_nets():
    for m, seed in ((14, 1), (14, 2), (20, 3)):
        net = S.make_random_net_k(m, "1 bis 5", seed)
        v = N.coalition_values(net)
        i = N.min_cut_edges(net)[0]
        a = K.max_payment_by_row_generation(net, i, "enum", v=v)
        b = K.max_payment_by_row_generation(net, i, "milp")
        assert a.status == b.status == "optimal" and a.x[i] == pytest.approx(b.x[i], abs=1e-6)
    assert "Aufzählungs- und gemischt-ganzzahlige Trennung: gleicher Zielwert in 3 von 3 Netzen (m = 14, 14, 20)" in README


def test_row_generation_needs_far_fewer_rows_than_coalitions():
    net = S.make_random_net_k(30, "1 bis 5", 1, n=C.BIG_NODES)
    res = K.max_payment_by_row_generation(net, N.min_cut_edges(net)[0], "milp")
    assert res.status == "optimal" and res.rows < 200 < 2 ** 30 - 2


# ----------------------------------------------------------------------------------------------------------------------- Rechengrenzen
def test_slider_maxima_and_limits_named_in_the_readme():
    assert (C.K_MIN, C.K_MAX, C.K_BIG_MIN, C.K_BIG_MAX, K.CORE_LP_MAX_PLAYERS) == (2, 20, 10, 40, 14)
    for needle in ("k von 2 bis 20", "10 bis 40 Kanten", "bis 14 Kanten", "bis 12 Kanten ohne Knopf", "bis 10 Kanten", "bis 12 Kanten zusätzlich als Brüche", "(bis 20 Kanten)"):
        assert needle in README, needle
    assert C.RANGES_LIVE_MAX == 12 and C.AUSBAU_MAX == 10 and C.CONVEX_CHECK_MAX == 14 and K.SEPARATION_ENUM_MAX == 20 and G.EXACT_SHAPLEY_MAX_PLAYERS == 12


def test_hypothesis_table_names_each_h_once():
    for h in range(1, 12):
        assert f"| H{h} |" in README, h


def test_every_orders_enumeration_count_in_the_oracle_note():
    assert len(list(itertools.permutations(range(6)))) == 720                    # Netz mit sechs Kanten (größtes im Orakel-Lauf): 720 Reihenfolgen
    assert "720 Reihenfolgen" in README
