"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.

Das Netzbild: Knoten in Schichten (längster Weg von der Quelle), Kanten als Pfeile in der Farbe ihres Betriebs, Breite nach Kapazität,
rote Unterlage für die Kanten eines minimalen Schnitts, Beschriftung als Annotation mit Hintergrund (nicht als Text-Spur)."""

import math

import plotly.graph_objects as go

import fsp_constants as C
import fsp_net as N


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def owner_color(i: int) -> str:
    return C.OWNER_COLORS[i % len(C.OWNER_COLORS)]


def layout_nodes(net: N.Net) -> dict:
    """Knoten -> (x, y): x = Schicht (längster Weg von der Quelle, Senke ganz rechts), y gleichmäßig je Schicht um 0 verteilt."""
    layer = {v: 0 for v in range(net.n)}
    for v in N.topological_order(net):
        for a, b, _ in net.edges:
            if a == v:
                layer[b] = max(layer[b], layer[v] + 1)
    last = max(layer.values())
    layer[net.sink] = last              # die Senke steht ganz rechts; Knoten ohne Kante bleiben in Schicht 0
    groups: dict = {}
    for v in range(net.n):
        groups.setdefault(layer[v], []).append(v)
    pos = {}
    for x, nodes in groups.items():
        count = len(nodes)
        for j, v in enumerate(nodes):
            pos[v] = (float(x), (j - (count - 1) / 2) * 1.2)
    return pos


def build_net(net: N.Net, flows=None, cut_edges=(), highlight=()) -> go.Figure:
    """Netzbild. `flows`: Fluss je Kante (Beschriftung „A: 3/4“), `cut_edges`: Kanten eines minimalen Schnitts (rote Unterlage), `highlight`: Kanten in kräftigem Rand."""
    pos = layout_nodes(net)
    names = N.edge_names(net.m)
    fig = go.Figure()
    seen: dict = {}
    mids_x, mids_y, hover = [], [], []
    ys = [p[1] for p in pos.values()]
    for i, (a, b, c) in enumerate(net.edges):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        idx = seen.get((a, b), 0)
        seen[(a, b)] = idx + 1
        same = sum(1 for e in net.edges if (e[0], e[1]) == (a, b))
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy) or 1.0
        off = (idx - (same - 1) / 2) * 0.28
        ox, oy = -dy / norm * off, dx / norm * off
        sx, sy, ex, ey = x0 + ox, y0 + oy, x1 + ox, y1 + oy
        width = 1.5 + min(c, 10) * 0.7
        if i in cut_edges:
            fig.add_trace(go.Scatter(x=[sx, ex], y=[sy, ey], mode="lines", line=dict(color=C.COLOR_CUT, width=width + 7), opacity=0.35, hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=ex, y=ey, ax=sx, ay=sy, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=width,
                           arrowcolor=owner_color(i), standoff=14, startstandoff=14)
        label = f"{names[i]}: {flows[i]}/{c}" if flows is not None else f"{names[i]}: {c}"
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        fig.add_annotation(x=mx, y=my, text=f"<b>{label}</b>", showarrow=False, bgcolor="rgba(255,255,255,0.88)", font=dict(size=12, color="#222"),
                           bordercolor=owner_color(i) if i in highlight else "rgba(0,0,0,0)", borderwidth=2, borderpad=2)
        mids_x.append(mx)
        mids_y.append(my)
        hover.append(f"Betrieb {names[i]}: Kapazität {c}" + (f", Fluss {flows[i]}" if flows is not None else "") + (" (im minimalen Schnitt)" if i in cut_edges else ""))
    fig.add_trace(go.Scatter(x=mids_x, y=mids_y, mode="markers", marker=dict(size=14, opacity=0), hovertext=hover, hoverinfo="text", showlegend=False))
    labels = {0: "s", net.sink: "t"}
    fig.add_trace(go.Scatter(x=[pos[v][0] for v in range(net.n)], y=[pos[v][1] for v in range(net.n)], mode="markers",
                             marker=dict(size=30, color="white", line=dict(color="#555", width=2)), hoverinfo="skip", showlegend=False))
    for v in range(net.n):
        fig.add_annotation(x=pos[v][0], y=pos[v][1], text=f"<b>{labels.get(v, v)}</b>", showarrow=False, font=dict(size=14, color="#222"))
    xs = [p[0] for p in pos.values()]
    fig.update_xaxes(visible=False, range=[min(xs) - 0.5, max(xs) + 0.5])
    fig.update_yaxes(visible=False, range=[min(ys) - 0.9, max(ys) + 0.9])
    height = int(min(520, max(240, 90 * (max(ys) - min(ys) + 2.2))))
    return _base(fig, height=height)


def build_ranges(net: N.Net, ranges, phi) -> go.Figure:
    """Spielraum im Kern je Kante: Balken von der Kern-Untergrenze bis zur Kern-Obergrenze, der Shapley-Wert als Raute."""
    names = N.edge_names(net.m)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[float(hi - lo) for lo, hi in ranges], base=[float(lo) for lo, _ in ranges], marker_color=[owner_color(i) for i in range(net.m)],
                         opacity=0.75, name="Spielraum im Kern", hovertemplate="%{x}: von %{base:.3f} bis %{customdata:.3f}<extra></extra>",
                         customdata=[float(hi) for _, hi in ranges]))
    fig.add_trace(go.Scatter(x=names, y=[float(p) for p in phi], mode="markers", marker=dict(symbol="diamond", size=11, color="#111", line=dict(color="white", width=1)),
                             name="Shapley-Wert", hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="Zahlung")
    fig.update_xaxes(title_text="Betrieb (Kante)", type="category")
    return _base(fig)


def build_e1(cells_by_label: dict) -> go.Figure:
    """E1: Anteil der Netze mit Shapley-Wert im Kern über der Kantenzahl; eine Linie je Beschriftung (Netzform und Kapazitätsmodus)."""
    fig = go.Figure()
    colors = [C.COLOR_LINE, C.COLOR_LINE2, "#54a24b", "#b279a2", "#9d755d", "#e45756"]
    for n, (label, cells) in enumerate(cells_by_label.items()):
        cells = sorted(cells, key=lambda c: c["k"])
        fig.add_trace(go.Scatter(x=[c["k"] for c in cells], y=[100 * c["in_core_mean"] for c in cells], mode="lines+markers", name=label,
                                 line=dict(color=colors[n % len(colors)], width=2.5), hovertemplate="k = %{x}: %{y:.1f} %<extra>" + label + "</extra>"))
    fig.update_xaxes(title_text="Zahl der Kanten k")
    fig.update_yaxes(title_text="Shapley im Kern (% der Netze)", range=[-3, 103])
    return _base(fig)


def build_ausbau(result: dict, edge_name: str) -> go.Figure:
    """E3: Shapley-Wert, Kern-Untergrenze und Kern-Obergrenze einer Kante über ihrer Kapazität."""
    caps = result["caps"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=caps, y=result["shapley"], mode="lines+markers", name="Shapley-Wert", line=dict(color="#111", width=3)))
    fig.add_trace(go.Scatter(x=caps, y=result["core_min"], mode="lines+markers", name="Kern-Untergrenze", line=dict(color=C.COLOR_LINE, width=2.5)))
    fig.add_trace(go.Scatter(x=caps, y=result["core_max"], mode="lines+markers", name="Kern-Obergrenze", line=dict(color=C.COLOR_LINE2, width=2.5, dash="dot")))
    fig.update_xaxes(title_text=f"Kapazität der Kante {edge_name}", dtick=1)
    fig.update_yaxes(title_text="Zahlung an die Kante")
    return _base(fig)
