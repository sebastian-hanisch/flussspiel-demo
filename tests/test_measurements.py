"""Vorgerechnete Messreihen (data/messreihen.json): Hash der Quelltexte, innere Stimmigkeit, Neuberechnung kleiner Zellen gegen die Datei.

Die Datei entsteht aus `python tools/make_messreihen.py`; wer Kern, Netzerzeuger, Zufallsquelle oder Messroutinen ändert, rechnet sie neu und zieht die Zahlen in
README und tests/test_claims.py nach. Neu gerechnet wird hier nur, was auf jeder Plattform gleich ausfällt: Anteile aus Zählungen mit Toleranz 1e-9 über Netze aus
SplitMix64 (Ganzzahlen); Laufzeiten und Zeilenzahlen der Programme (HiGHS-Wahl bei Gleichständen) stehen nur als Band im Test."""

import pytest

import fsp_constants as C
import fsp_experiments as E
import fsp_tables as T


@pytest.fixture(scope="module")
def data():
    return T.load()


def test_the_file_was_made_from_the_current_sources(data):
    assert T.is_current(data), "data/messreihen.json stammt von einem älteren Stand des Codes: python tools/make_messreihen.py"
    assert data["meta"]["sweep_seeds"] == list(C.SWEEP_SEEDS)


def test_the_source_hash_ignores_line_endings(tmp_path, monkeypatch):
    before = T.source_hash()
    real = T.ROOT
    monkeypatch.setattr(T, "ROOT", tmp_path)
    for name in T.SOURCES:
        (tmp_path / name).write_bytes((real / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert T.source_hash() == before


def test_every_cell_is_complete_and_consistent(data):
    expected = {("Zufalls-DAG", c, k) for c in ("alle 1", "1 bis 5") for k in T.E1_DAG_KS + T.E1_DAG_BIG}
    expected |= {(f, c, k) for f in ("Reihen-Parallel-Netz", "Engpasskette") for c in ("alle 1", "1 bis 5") for k in T.E1_OTHER_KS}
    assert {(c["form"], c["capmode"], c["k"]) for c in data["e1"]} == expected
    for c in data["e1"]:
        runs = len(C.SWEEP_SEEDS) * (T.E1_BIG_RUNS if c["k"] in T.E1_DAG_BIG else T.E1_RUNS)
        assert c["runs"] == runs and 0 <= c["in_core_mean"] <= 1 and 0 <= c["surplus_share"] <= 1
        assert c["clean"] + round(c["surplus_share"] * runs) == runs and c["clean_in_core"] <= c["clean"]
        assert c["surplus_and_in_core"] == 0                      # Satz: überzählige Kante => Shapley außerhalb (Beweis in der App, Test in test_properties)


def test_the_share_with_shapley_in_the_core_is_made_of_clean_nets_only(data):
    """Im Kern liegt der Shapley-Wert nur in Netzen ohne überzählige Kante: Zählung je Zelle (Mittel über Seeds weicht von der Gesamtzählung nur durch gleich große Seeds ab)."""
    for c in data["e1"]:
        assert c["clean_in_core"] / c["runs"] == pytest.approx(c["in_core_mean"], abs=1e-12)


def test_recomputed_small_cells_equal_the_file(data):
    for k in (3, 4, 5):
        for capmode in ("alle 1", "1 bis 5"):
            stored = T.e1_cell(data, "Zufalls-DAG", capmode, k)
            fresh = E.shapley_core_cell("Zufalls-DAG", capmode, k, T.E1_RUNS)
            assert fresh == stored, (capmode, k)


def test_recomputed_other_forms_equal_the_file(data):
    for form in ("Reihen-Parallel-Netz", "Engpasskette"):
        stored = T.e1_cell(data, form, "alle 1", 4)
        assert E.shapley_core_cell(form, "alle 1", 4, T.E1_RUNS) == stored


def test_recomputed_hull_class_and_restfall_slices_equal_the_file(data):
    one = E.hull_cell("alle 1", (4,), 2, seeds=C.SWEEP_SEEDS[:1])
    assert one["equal_mean"] == 1.0
    assert E.restfall_table((3,), 5, seeds=C.SWEEP_SEEDS[:1])["nets"] == 5
    rf = data["restfall"]
    assert rf["convex_in_core"] == rf["convex"] and rf["clean_outside_nonconvex"] == rf["clean_outside"] and rf["surplus_and_in_core"] == 0


def test_rowgen_rows_stay_in_a_band_and_every_run_is_optimal(data):
    for r in data["e4"]:
        assert r["all_optimal"] and r["rows_min"] <= r["rows_mean"] <= r["rows_max"] and r["target_zero"] == 0
        assert r["rows_max"] < r["coalitions"] - 2 and r["rows_max"] <= 200
        if r["m"] <= 14:
            assert r["equal_to_full"] == r["runs"]
    fresh = E.rowgen_cell(10, 3, "enum", compare_full=True)
    assert fresh["all_optimal"] and fresh["equal_to_full"] == 3 and fresh["rows_max"] <= 40


def test_class_rows_follow_the_theorems(data):
    for r in data["e2"]["classes"]:
        assert r["surplus"]["max_mean"] == 0 and r["surplus"]["max_is_zero"] == r["surplus"]["edges"]       # überzählig: Kern-Obergrenze 0
        assert r["veto"]["max_mean"] == pytest.approx(1.0)                                                  # Veto: ganzer Wert


def test_hull_equals_core_for_unit_capacities_in_every_net_and_not_for_unequal_ones(data):
    by = {r["capmode"]: r for r in data["e2"]["hull"]}
    assert by["alle 1"]["equal_mean"] == 1.0 and by["alle 1"]["equal_std"] == 0.0
    assert by["1 bis 5"]["equal_mean"] < 1 and by["1 bis 20"]["equal_mean"] < 1
