"""Eigenschaftstests (hypothesis, feste Reihenfolge der Beispiele): Sätze, die für jedes Netz gelten, an zufälligen kleinen Netzen mit Nullspielern, Kapazität 0
und parallelen Kanten. Keine Mutationstests (Nutzerentscheidung 2026-10-28); die Sätze sind die Eigenschaften aus dem Plan (H1, H2, H4, H8, H9, H10)."""

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import fsp_core as K  # noqa: E402
import fsp_game as G  # noqa: E402
import fsp_net as N  # noqa: E402

PROFILE = settings(max_examples=60, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def nets(draw, max_edges=6, max_cap=4):
    n = draw(st.integers(2, 5))
    m = draw(st.integers(1, max_edges))
    edges = []
    for _ in range(m):
        a = draw(st.integers(0, n - 2))
        b = draw(st.integers(a + 1, n - 1))
        edges.append((a, b, draw(st.integers(0, max_cap))))
    return N.Net(n, tuple(edges), "hypothesis")


@PROFILE
@given(nets())
def test_shapley_is_efficient_and_null_players_get_nothing(net):
    v = N.coalition_values(net)
    phi = G.shapley(v)
    assert phi.sum() == pytest.approx(v[-1], abs=1e-9)
    for i, is_null in enumerate(G.null_players(v)):
        if is_null:
            assert phi[i] == pytest.approx(0, abs=1e-12)
            lo, hi = K.core_range(v, i)
            assert lo == pytest.approx(0, abs=K.TOL) and hi == pytest.approx(0, abs=K.TOL)


@PROFILE
@given(nets())
def test_the_core_is_never_empty_and_contains_the_minimal_cut_payoffs(net):
    v = N.coalition_values(net)
    assert K.core_nonempty(v)
    for x in K.min_cut_payoffs(net, v[-1]):
        assert G.in_core(v, x)


@PROFILE
@given(nets())
def test_a_surplus_edge_gets_zero_in_the_core_and_pushes_shapley_out(net):
    v = N.coalition_values(net)
    phi = G.shapley(v)
    for i in G.surplus_edges(v):
        assert K.core_range(v, i)[1] == pytest.approx(0, abs=K.TOL)
        assert phi[i] > 0
        assert not G.in_core(v, phi)


@PROFILE
@given(nets())
def test_a_veto_edge_can_claim_the_whole_value(net):
    v = N.coalition_values(net)
    for i in G.veto_edges(v):
        x = np.zeros(net.m)
        x[i] = v[-1]
        assert G.in_core(v, x)


@PROFILE
@given(nets())
def test_convex_games_have_the_shapley_value_in_the_core(net):
    v = N.coalition_values(net)
    if G.is_convex(v):
        assert G.in_core(v, G.shapley(v))


@PROFILE
@given(nets(max_edges=5), st.integers(0, 4))
def test_shapley_value_of_an_edge_never_falls_when_its_capacity_rises(net, which):
    i = which % net.m
    phis = [G.shapley(N.coalition_values(net.with_capacity(i, c)))[i] for c in range(0, 6)]
    assert np.diff(phis).min() >= -1e-9


@PROFILE
@given(nets(), st.data())
def test_cut_formula_equals_the_max_flow_on_a_random_coalition(net, data):
    v = N.coalition_values(net)
    mask = data.draw(st.integers(0, net.full))
    assert v[mask] == N.max_flow_value(net, mask)


@PROFILE
@given(nets())
def test_coalition_values_are_monotone_and_bounded_by_the_grand_value(net):
    v = N.coalition_values(net)
    idx = np.arange(len(v))
    for i in range(net.m):
        without = idx[(idx >> i & 1) == 0]
        assert np.all(v[without | (1 << i)] >= v[without] - 1e-12)
    assert v.max() == v[-1] and v[0] == 0
