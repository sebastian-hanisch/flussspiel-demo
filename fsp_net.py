"""Netz und Koalitionswerte des Flussspiels.

Ein gerichtetes Netz mit Quelle 0 und Senke n-1; jede Kante gehört genau einem Betrieb (Spieler = Kante) und hat eine ganzzahlige
Kapazität. Der Wert einer Koalition S ist der größte Fluss, den die Kanten aus S allein durchsetzen. Bit i der Koalitionsmaske ist Kante i.

Zwei unabhängige Rechenwege für v(S):
  * `max_flow`: kürzeste Erweiterungspfade (Edmonds-Karp), für eine einzelne Koalition und für die Flussanzeige;
  * `coalition_values`: Max-Flow-Min-Cut, v(S) = min über alle s-t-Schnitte C von c(C ∩ S), vektorisiert über alle 2^m Koalitionen.
Beide sind in `tests/test_oracle_fluss.py` gegeneinander und gegen networkx und scipy geprüft."""

from collections import deque
from dataclasses import dataclass

import numpy as np

SOURCE = 0
MAX_INNER_NODES_FOR_CUTS = 16   # 2^16 Knotenteilmengen: Obergrenze der Schnittaufzählung (die App bleibt bei höchstens 10 inneren Knoten)


@dataclass(frozen=True)
class Net:
    """n Knoten (0 = Quelle, n-1 = Senke), Kanten (von, nach, Kapazität)."""

    n: int
    edges: tuple
    name: str = ""

    @property
    def m(self) -> int:
        return len(self.edges)

    @property
    def sink(self) -> int:
        return self.n - 1

    @property
    def caps(self) -> tuple:
        return tuple(c for _, _, c in self.edges)

    @property
    def full(self) -> int:
        """Maske der großen Koalition."""
        return (1 << self.m) - 1

    def with_capacity(self, i: int, c: int) -> "Net":
        edges = tuple((a, b, c if j == i else cc) for j, (a, b, cc) in enumerate(self.edges))
        return Net(self.n, edges, self.name)


def check_net(net: Net) -> None:
    """Wirft ValueError bei einem unzulässigen Netz: n >= 2, Knoten im Bereich, keine Schleifen, ganzzahlige Kapazitäten >= 0, keine Kreise."""
    if net.n < 2:
        raise ValueError("ein Netz braucht mindestens Quelle und Senke")
    for a, b, c in net.edges:
        if not (0 <= a < net.n and 0 <= b < net.n):
            raise ValueError(f"Knoten außerhalb des Netzes: {(a, b)}")
        if a == b:
            raise ValueError("Schleifen sind nicht erlaubt")
        if int(c) != c or c < 0:
            raise ValueError("Kapazitäten sind ganze Zahlen >= 0")
    topological_order(net)


def topological_order(net: Net) -> list:
    """Knotenreihenfolge, in der jede Kante vorwärts zeigt; ValueError bei einem Kreis."""
    indeg = [0] * net.n
    out = [[] for _ in range(net.n)]
    for a, b, _ in net.edges:
        indeg[b] += 1
        out[a].append(b)
    queue = deque(v for v in range(net.n) if indeg[v] == 0)
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for w in out[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    if len(order) != net.n:
        raise ValueError("das Netz enthält einen Kreis (nur Netze ohne Kreise)")
    return order


def relevant_edges(net: Net) -> list:
    """Indizes der Kanten auf mindestens einem s-t-Weg (Anfang von der Quelle erreichbar, Ende erreicht die Senke)."""
    n = net.n
    fwd = [[] for _ in range(n)]
    bwd = [[] for _ in range(n)]
    for a, b, _ in net.edges:
        fwd[a].append(b)
        bwd[b].append(a)

    def reach(start, adj):
        seen = {start}
        stack = [start]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen

    from_s, to_t = reach(SOURCE, fwd), reach(net.sink, bwd)
    return [i for i, (a, b, _) in enumerate(net.edges) if a in from_s and b in to_t]


# ----------------------------------------------------------------------------------------------------------- Fluss einer Koalition
def max_flow(net: Net, mask: int | None = None) -> tuple:
    """Größter Fluss über die Kanten aus `mask` (Standard: alle). Rückgabe: (Wert, Fluss je Kante als Liste ganzer Zahlen).

    Edmonds-Karp auf dem Restnetz; parallele Kanten bleiben getrennte Bögen, damit der Fluss je Kante (je Betrieb) ablesbar ist."""
    if mask is None:
        mask = net.full
    m = net.m
    res = [0] * (2 * m)
    adj = [[] for _ in range(net.n)]
    for i, (a, b, c) in enumerate(net.edges):
        if mask >> i & 1:
            res[2 * i] = int(c)
        adj[a].append(2 * i)       # Vorwärtsbogen 2i: a -> b
        adj[b].append(2 * i + 1)   # Rückbogen 2i+1: b -> a
    head = [0] * (2 * m)
    for i, (a, b, _) in enumerate(net.edges):
        head[2 * i], head[2 * i + 1] = b, a
    value = 0
    sink = net.sink
    while True:
        parent = [-1] * net.n
        parent[SOURCE] = -2
        queue = deque([SOURCE])
        while queue and parent[sink] == -1:
            v = queue.popleft()
            for arc in adj[v]:
                w = head[arc]
                if res[arc] > 0 and parent[w] == -1:
                    parent[w] = arc
                    queue.append(w)
        if parent[sink] == -1:
            break
        push = None
        v = sink
        while v != SOURCE:
            arc = parent[v]
            push = res[arc] if push is None else min(push, res[arc])
            v = head[arc ^ 1]
        v = sink
        while v != SOURCE:
            arc = parent[v]
            res[arc] -= push
            res[arc ^ 1] += push
            v = head[arc ^ 1]
        value += push
    flows = [res[2 * i + 1] if mask >> i & 1 else 0 for i in range(m)]
    return value, flows


def max_flow_value(net: Net, mask: int | None = None) -> int:
    return max_flow(net, mask)[0]


def min_cut_edges(net: Net) -> list:
    """Kanten eines minimalen Schnitts: nach einem größten Fluss die Kanten von der Quelle aus erreichbarer Knoten zu nicht erreichbaren (Restnetz).

    Der Schnitt hat die Kapazität des größten Flusses; es ist einer von möglicherweise mehreren (der der Quelle nächste)."""
    _, flows = max_flow(net)
    reach = {SOURCE}
    stack = [SOURCE]
    while stack:
        v = stack.pop()
        for (a, b, c), f in zip(net.edges, flows):
            if a == v and b not in reach and c - f > 0:
                reach.add(b)
                stack.append(b)
            if b == v and a not in reach and f > 0:
                reach.add(a)
                stack.append(a)
    return [i for i, (a, b, _) in enumerate(net.edges) if a in reach and b not in reach]


# ---------------------------------------------------------------------------------------------------------------------- Schnitte
def cut_matrix(net: Net) -> np.ndarray:
    """Bool-Matrix (Schnitt x Kante): Kante i liegt im Schnitt, wenn ihr Anfang auf der Quellenseite und ihr Ende auf der Senkenseite liegt.

    Aufgezählt werden alle 2^(n-2) Teilmengen der inneren Knoten (Quelle immer drin, Senke nie); doppelte Zeilen sind entfernt."""
    inner = net.n - 2
    if inner > MAX_INNER_NODES_FOR_CUTS:
        raise ValueError(f"zu viele innere Knoten für die Schnittaufzählung ({inner} > {MAX_INNER_NODES_FOR_CUTS})")
    subsets = np.arange(1 << inner, dtype=np.int64)
    side = np.zeros((1 << inner, net.n), dtype=bool)
    side[:, SOURCE] = True
    for node in range(1, net.n - 1):
        side[:, node] = (subsets >> (node - 1)) & 1 == 1
    a = np.array([e[0] for e in net.edges], dtype=np.int64)
    b = np.array([e[1] for e in net.edges], dtype=np.int64)
    if net.m == 0:
        return np.zeros((1, 0), dtype=bool)
    crossing = side[:, a] & ~side[:, b]
    return np.unique(crossing, axis=0)


def cut_vectors(net: Net, minimal_only: bool = False) -> np.ndarray:
    """Kapazitätsvektoren aller s-t-Schnitte (Zeile = Schnitt, Spalte = Kante).

    `minimal_only=True` lässt jeden Schnitt weg, dessen Vektor einen anderen Schnitt in allen Kanten erreicht oder übertrifft: das Minimum
    über die Schnitte ändert sich dadurch nicht, die Zeilenzahl sinkt stark."""
    caps = np.array(net.caps, dtype=float)
    w = cut_matrix(net).astype(float) * caps[None, :]
    w = np.unique(w, axis=0)
    if not minimal_only or len(w) <= 1:
        return w
    keep = np.ones(len(w), dtype=bool)
    for i in range(len(w)):
        dominated = np.all(w <= w[i][None, :], axis=1) & np.any(w < w[i][None, :], axis=1)
        if dominated.any():
            keep[i] = False
    return w[keep]


def additive_array(x) -> np.ndarray:
    """x(S) für alle 2^m Koalitionen S; Bit i der Maske ist Eintrag i."""
    arr = np.zeros(1)
    for xi in x:
        arr = np.concatenate([arr, arr + xi])
    return arr


def coalition_values(net: Net) -> np.ndarray:
    """v(S) für alle 2^m Koalitionen über die Schnittformel (ganze Zahlen als Gleitkommazahlen, exakt)."""
    if net.m == 0:
        return np.zeros(1)
    v = None
    for w in cut_vectors(net, minimal_only=True):
        a = additive_array(w)
        v = a if v is None else np.minimum(v, a)
    return v


def minimal_cut_vectors(net: Net, v_grand: float | None = None) -> np.ndarray:
    """Auszahlungsvektoren der minimalen Schnitte: x_e = c_e auf den Kanten eines Schnitts kleinster Kapazität, sonst 0."""
    w = cut_vectors(net)
    if v_grand is None:
        v_grand = float(w.sum(axis=1).min()) if len(w) else 0.0
    return w[np.isclose(w.sum(axis=1), v_grand, rtol=0, atol=1e-9)]


def edge_names(m: int) -> list:
    """Betriebsnamen A, B, ... Z, A2, B2, ... (Spieler = Kante)."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [letters[i % 26] + ("" if i < 26 else str(i // 26 + 1)) for i in range(m)]


def coalition_label(mask: int, m: int) -> str:
    names = edge_names(m)
    inside = [names[i] for i in range(m) if mask >> i & 1]
    return "{" + ", ".join(inside) + "}" if inside else "{}"
