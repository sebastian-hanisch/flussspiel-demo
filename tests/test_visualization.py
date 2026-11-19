"""Abbildungen: Netzbild (Schichten, parallele Kanten, Beschriftung), Achsen gesperrt, Balken, Linien."""

import fsp_constants as C
import fsp_net as N
import fsp_scenario as S
import fsp_visualization as V


def _locked(fig):
    return all(ax.fixedrange for ax in (fig.layout.xaxis, fig.layout.yaxis))


def test_layout_puts_source_left_and_sink_right_in_layers():
    for name in C.EXAMPLES:
        net = S.example_net(name)
        xs = {v: p[0] for v, p in V.layout_nodes(net).items()}
        assert xs[0] == min(xs.values()) and xs[net.sink] == max(xs.values())
        assert all(xs[a] < xs[b] for a, b, _ in net.edges), name                # jede Kante zeigt nach rechts


def test_series_net_has_one_layer_per_step(reihe):
    assert [V.layout_nodes(reihe)[v][0] for v in range(4)] == [0.0, 1.0, 2.0, 3.0]


def test_net_figure_labels_every_edge_with_capacity_and_flow(drei):
    _, flows = N.max_flow(drei)
    fig = V.build_net(drei, flows=flows, cut_edges=[0])
    texts = " ".join(a.text or "" for a in fig.layout.annotations)
    for label in ("A: 4/4", "B: 3/3", "C: 1/5"):
        assert label in texts
    assert _locked(fig)
    assert any(t.line is not None and t.line.color == C.COLOR_CUT for t in fig.data)        # rote Unterlage für den Schnitt


def test_net_figure_without_flow_shows_capacities_only(drei):
    texts = " ".join(a.text or "" for a in V.build_net(drei).layout.annotations)
    assert "A: 4" in texts and "A: 4/" not in texts


def test_parallel_edges_do_not_overlap(parallel):
    arrows = [(a.x, a.y, a.ax, a.ay) for a in V.build_net(parallel).layout.annotations if a.showarrow]
    assert len(arrows) == 2 and arrows[0] != arrows[1] and arrows[0][1] != arrows[1][1]


def test_net_figure_for_every_random_form_and_the_largest_net():
    for form in C.RANDOM_FORMS:
        net = S.make_net(form, C.K_MAX, "1 bis 20", 3)
        fig = V.build_net(net, flows=N.max_flow(net)[1], cut_edges=N.min_cut_edges(net))
        assert _locked(fig) and len([a for a in fig.layout.annotations if a.showarrow]) == C.K_MAX


def test_ranges_figure_has_one_bar_and_one_marker_per_edge(drei):
    fig = V.build_ranges(drei, [[3, 4], [0, 0], [0, 1]], [2.5, 0.5, 1.0])
    assert len(fig.data[0].x) == 3 and len(fig.data[1].x) == 3 and _locked(fig)
    assert list(fig.data[0].base) == [3.0, 0.0, 0.0] and list(fig.data[0].y) == [1.0, 0.0, 1.0]


def test_e1_and_ausbau_figures():
    fig = V.build_e1({"a": [{"k": 4, "in_core_mean": 0.3}, {"k": 3, "in_core_mean": 0.6}]})
    assert list(fig.data[0].x) == [3, 4] and list(fig.data[0].y) == [60.0, 30.0] and _locked(fig)
    res = {"caps": [1, 2], "shapley": [1.0, 2.0], "core_min": [0.0, 0.0], "core_max": [1.0, 2.0]}
    assert len(V.build_ausbau(res, "A").data) == 3
