"""Netze: Handbeispiele, Zufallsnetze (genau k Kanten, alle auf s-t-Wegen), Zufalls-Seed, Hash-Unabhängigkeit, SplitMix64-Kopie."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fsp_constants as C
import fsp_net as N
import fsp_rng
import fsp_scenario as S

ROOT = Path(__file__).resolve().parent.parent
RNG_SOURCE = ("ab-test-demo", "abt_rng.py")


# ----------------------------------------------------------------------------------------------------------------------- SplitMix64
def test_rng_copy_is_verbatim_apart_from_the_header():
    folder, original = RNG_SOURCE
    source = ROOT.parent / folder / original
    mine = (ROOT / "fsp_rng.py").read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_copied_generator_has_the_documented_stream():
    """SplitMix64 mit Seed 0: die bekannten ersten Werte der Konvention (Vigna)."""
    rng = fsp_rng.SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_block_equals_the_scalar_stream():
    for seed in (0, 1, 35, 2 ** 63 + 5):
        rng = fsp_rng.SplitMix64(seed)
        assert np.array_equal(fsp_rng.uniform_block(seed, 50), np.array([rng.uniform() for _ in range(50)]))


def test_rng_module_has_no_other_imports_than_numpy_and_math():
    tree = ast.parse((ROOT / "fsp_rng.py").read_text(encoding="utf-8"))
    assert {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} == {"math", "numpy"}


def test_no_module_uses_the_random_modules_of_python_or_numpy():
    """Kein `random` und kein `numpy.random` für Befunde: aller Zufall läuft über SplitMix64. Ausnahme `fsp_presets.py`: der Knopf „Neue Instanz“ würfelt
    einen neuen Zufalls-Seed der Oberfläche (kein Befund; der Seed selbst steht danach in der Adresszeile)."""
    for path in ROOT.glob("*.py"):
        if path.name == "fsp_presets.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert "random" not in [a.name for a in node.names], path.name
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ("random", "numpy.random"), path.name
            if isinstance(node, ast.Attribute) and node.attr == "random" and isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy"):
                raise AssertionError(f"numpy.random in {path.name}")


def test_randint_covers_the_closed_range():
    rng = fsp_rng.SplitMix64(5)
    draws = [S.randint(rng, 1, 5) for _ in range(500)]
    assert min(draws) == 1 and max(draws) == 5 and set(draws) == {1, 2, 3, 4, 5}
    assert all(S.randint(rng, 3, 3) == 3 for _ in range(10))


# ------------------------------------------------------------------------------------------------------------------- Handbeispiele
@pytest.mark.parametrize("name", C.EXAMPLES)
def test_examples_are_valid_acyclic_nets_with_every_edge_on_a_path(name):
    net = S.example_net(name)
    N.check_net(net)
    assert N.relevant_edges(net) == list(range(net.m))


def test_example_values_by_hand():
    expected = {"Reihe (5, 2, 3)": (3, 2), "Parallel (2, 3)": (2, 5), "Drei Kanten (4 | 3, 5)": (3, 4), "Brücke mit Einheitskapazität": (5, 2),
                "Brücke (3, 2, 1, 2, 3)": (5, 5), "Vier Kanten (5, 1 | 3, 5)": (4, 6)}
    assert set(expected) == set(C.EXAMPLES)
    for name, (edges, value) in expected.items():
        net = S.example_net(name)
        assert net.m == edges and N.max_flow_value(net) == value, name


def test_unknown_names_are_rejected():
    with pytest.raises(ValueError):
        S.example_net("gibt es nicht")
    with pytest.raises(ValueError):
        S.make_net("gibt es nicht", 5, "alle 1", 1)


def test_examples_ignore_k_capacity_mode_and_seed():
    a = S.make_net("Drei Kanten (4 | 3, 5)", 2, "alle 1", 0)
    b = S.make_net("Drei Kanten (4 | 3, 5)", 20, "1 bis 20", 999)
    assert a == b


# ----------------------------------------------------------------------------------------------------------------------- Zufallsnetze
@pytest.mark.parametrize("form", C.RANDOM_FORMS)
def test_random_nets_have_exactly_k_edges_all_on_a_path_for_every_k(form):
    for k in range(C.K_MIN, C.K_MAX + 1):
        for seed in (1, 2, 3):
            net = S.make_net(form, k, "1 bis 5", seed)
            assert net.m == k, (form, k, seed)
            assert N.relevant_edges(net) == list(range(k)), (form, k, seed)
            assert net.n - 2 <= 10 and all(1 <= c <= 5 for c in net.caps)


@pytest.mark.parametrize("capmode", list(C.CAPMODES))
def test_capacity_modes_draw_from_their_range(capmode):
    lo, hi = C.CAPMODES[capmode]
    caps = [c for seed in range(30) for c in S.make_net("Engpasskette", 8, capmode, seed).caps]
    assert min(caps) >= lo and max(caps) <= hi
    if lo != hi:
        assert min(caps) == lo and max(caps) == hi


def test_unit_mode_makes_every_capacity_one():
    for form in C.RANDOM_FORMS:
        assert set(S.make_net(form, 12, "alle 1", 4).caps) == {1}


def test_same_seed_same_net_and_other_seed_other_net():
    for form in C.RANDOM_FORMS:
        a, b, c = (S.make_net(form, 10, "1 bis 20", s) for s in (7, 7, 8))
        assert a == b and a != c


@pytest.mark.parametrize("form", C.RANDOM_FORMS)
def test_random_nets_are_acyclic_with_source_zero_and_sink_last(form):
    for seed in range(1, 8):
        net = S.make_net(form, 9, "1 bis 5", seed)
        N.check_net(net)
        order = N.topological_order(net)
        assert order[0] == 0 and order[-1] == net.sink
        assert all(a < b for a, b, _ in net.edges), "normalize nummeriert topologisch: jede Kante zeigt zu einer größeren Nummer"


def test_the_dag_fill_in_for_missing_edges_is_rare():
    """Verfahren nach Plan; greift die Auffüllung durch Parallelkanten, steht das so im Test (Zählung über alle k und 5 Zufalls-Seeds)."""
    from fsp_rng import SplitMix64

    used = total = 0
    for k in range(C.K_MIN, C.K_MAX + 1):
        for seed in range(1, 6):
            _, fill = S.random_dag_with_info(SplitMix64(seed), k, "1 bis 5")
            used += fill
            total += 1
    assert total == 95 and used <= 5


def test_the_dag_fill_in_keeps_exactly_k_relevant_edges_even_with_one_attempt():
    from fsp_rng import SplitMix64

    net, fill = S.random_dag_with_info(SplitMix64(3), 18, "1 bis 5", n=5, max_attempts=1)
    assert net.m == 18 and N.relevant_edges(net) == list(range(18))
    assert fill or net.m == 18


def test_series_parallel_respects_the_inner_node_budget():
    from fsp_rng import SplitMix64

    for seed in range(1, 15):
        net = S.series_parallel(SplitMix64(seed), 20, "alle 1", max_inner=3)
        assert net.n - 2 <= 3 and net.m == 20


def test_bottleneck_chain_groups_are_in_series():
    net = S.make_net("Engpasskette", 9, "1 bis 5", 3)
    nodes = sorted({x for a, b, _ in net.edges for x in (a, b)})
    assert nodes == list(range(net.n)) and all(b == a + 1 for a, b, _ in net.edges)
    assert net.n >= 3                                 # mindestens zwei Gruppen (ein Engpass in Reihe)


def test_normalize_renumbers_topologically_and_drops_unused_nodes():
    net = S.normalize(6, [(0, 4, 2), (4, 5, 3), (0, 5, 1)], "x")
    assert net.n == 3 and net.edges == ((0, 1, 2), (1, 2, 3), (0, 2, 1))


def test_nodes_for_follows_the_measurement_series():
    assert [S.nodes_for(k) for k in (2, 10, 11, 16, 17, 20)] == [6, 6, 7, 7, 8, 8]


def test_make_random_net_k_accepts_big_nets():
    net = S.make_random_net_k(40, "1 bis 5", 3, n=C.BIG_NODES)
    assert net.m == 40 and net.n <= C.BIG_NODES and N.relevant_edges(net) == list(range(40))


# -------------------------------------------------------------------------------------------------------------- Hash-Unabhängigkeit
def test_nets_do_not_depend_on_the_hash_seed_of_the_process():
    code = (
        "import sys; sys.path.insert(0, %r);\n"
        "import fsp_scenario as S\n"
        "print([S.make_net(f, 9, '1 bis 5', 11).edges for f in ('Zufalls-DAG', 'Reihen-Parallel-Netz', 'Engpasskette')])"
    ) % str(ROOT)
    outs = []
    for hashseed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout)
    assert outs[0] == outs[1] and outs[0].strip()
