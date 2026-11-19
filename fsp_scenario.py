"""Netze des Flussspiels: Handbeispiele und Zufallsnetze (alle Kanten liegen auf s-t-Wegen).

Zufall nur über SplitMix64 (`fsp_rng`, wörtliche Kopie der Konvention): gleicher Zufalls-Seed, gleiches Netz, auf jeder Plattform.
Zufallsnetze enthalten nur Kanten auf s-t-Wegen: eine Kante außerhalb jedes Weges ist ein Nullspieler, bekommt in Shapley-Wert und Kern 0 und
würde jede Zählung über „Spieler“ aufblähen (die erste Vormessung zählte so 9 Kanten, von denen nur 1 bis 6 auf einem Weg lagen)."""

import fsp_constants as C
from fsp_net import Net, check_net, relevant_edges, topological_order
from fsp_rng import SplitMix64


def randint(rng: SplitMix64, lo: int, hi: int) -> int:
    """Gleichverteilte ganze Zahl in [lo, hi] (beide inklusive)."""
    return lo + int(rng.uniform() * (hi - lo + 1))


def draw_capacity(rng: SplitMix64, capmode: str) -> int:
    lo, hi = C.CAPMODES[capmode]
    return lo if lo == hi else randint(rng, lo, hi)


def normalize(n: int, edges, name: str = "") -> Net:
    """Knoten so umnummerieren, dass die Nummern die Topologie widerspiegeln, die Quelle 0 und die Senke n'-1 ist; unbenutzte Knoten entfallen.

    Voraussetzung: Quelle = 0, Senke = n-1 in der Eingabe, Netz ohne Kreise."""
    used = {x for a, b, _ in edges for x in (a, b)} | {0, n - 1}
    tmp = Net(n, tuple(edges))
    order = [v for v in topological_order(tmp) if v in used and v not in (0, n - 1)]
    mapping = {0: 0}
    for rank, v in enumerate(order, start=1):
        mapping[v] = rank
    mapping[n - 1] = len(order) + 1
    return Net(len(order) + 2, tuple((mapping[a], mapping[b], c) for a, b, c in edges), name)


# --------------------------------------------------------------------------------------------------------------------- Handbeispiele
def example_net(name: str) -> Net:
    """Die festen Beispielnetze (Namen in C.EXAMPLES)."""
    if name == "Reihe (5, 2, 3)":
        return Net(4, ((0, 1, 5), (1, 2, 2), (2, 3, 3)), name)
    if name == "Parallel (2, 3)":
        return Net(2, ((0, 1, 2), (0, 1, 3)), name)
    if name == "Drei Kanten (4 | 3, 5)":
        return Net(3, ((0, 1, 4), (1, 2, 3), (1, 2, 5)), name)
    if name == "Brücke mit Einheitskapazität":
        return wheatstone((1, 1, 1, 1, 1), name)
    if name == "Brücke (3, 2, 1, 2, 3)":
        return wheatstone((3, 2, 1, 2, 3), name)
    if name == "Vier Kanten (5, 1 | 3, 5)":
        return Net(3, ((0, 1, 5), (0, 1, 1), (1, 2, 3), (1, 2, 5)), name)
    raise ValueError(f"unbekanntes Beispielnetz: {name}")


def wheatstone(c, name: str = "Brücke") -> Net:
    """Brücke mit fünf Kanten: s->1, s->2, 1->2 (die Brücke), 1->t, 2->t; Knoten 0, 1, 2, 3 (= t)."""
    return Net(4, ((0, 1, c[0]), (0, 2, c[1]), (1, 2, c[2]), (1, 3, c[3]), (2, 3, c[4])), name)


# ------------------------------------------------------------------------------------------------------------------------ Zufallsnetze
def nodes_for(k: int) -> int:
    """Knotenzahl des Zufalls-DAG je Kantenzahl (wie in der Messreihe: 6 Knoten bis k = 10, 7 bis k = 16, danach 8)."""
    return 6 if k <= 10 else (7 if k <= 16 else 8)


def _bfs_relevant_count(n: int, edges) -> list:
    return relevant_edges(Net(n, tuple(edges)))


def random_dag_with_info(rng: SplitMix64, k: int, capmode: str, n: int | None = None, max_attempts: int = 400) -> tuple:
    """(Netz, Auffüllung genutzt?): Zufalls-DAG mit genau k Kanten, alle auf s-t-Wegen.

    Verfahren: Kanten (a < b, zwei verschiedene Knoten) einzeln ziehen und nach jeder Kante die Kanten an s-t-Wegen zählen (die übrigen Kanten
    bleiben in der Liste und können durch spätere Kanten auf einen Weg kommen; zurückgeschnitten wird nicht). Genau k Kanten an Wegen:
    fertig, das Netz besteht nur aus diesen; mehr als k: neuer Versuch. Nach `max_attempts` Versuchen ohne Treffer wird der beste Stand (weniger als k Kanten) durch
    parallele Kopien zufällig gewählter Kanten auf k aufgefüllt (selten; ein Test zählt die Fälle)."""
    n = nodes_for(k) if n is None else n
    best: list = []
    for _ in range(max_attempts):
        edges: list = []
        for _step in range(6 * k + 10):
            a = randint(rng, 0, n - 1)
            b = randint(rng, 0, n - 2)
            if b >= a:
                b += 1
            a, b = min(a, b), max(a, b)
            edges.append((a, b, draw_capacity(rng, capmode)))
            rel = _bfs_relevant_count(n, edges)
            if len(rel) == k:
                return normalize(n, [edges[i] for i in rel], "Zufalls-DAG"), False
            if len(rel) > k:
                break
            if len(rel) > len(best):
                best = [edges[i] for i in rel]
    if not best:
        best = [(0, n - 1, draw_capacity(rng, capmode))]
    while len(best) < k:
        a, b, c = best[randint(rng, 0, len(best) - 1)]
        best.append((a, b, c))
    return normalize(n, best, "Zufalls-DAG"), True


def random_dag(rng: SplitMix64, k: int, capmode: str, n: int | None = None, max_attempts: int = 400) -> Net:
    """Wie `random_dag_with_info`, nur das Netz."""
    return random_dag_with_info(rng, k, capmode, n, max_attempts)[0]


def series_parallel(rng: SplitMix64, k: int, capmode: str, max_inner: int = C.SP_MAX_INNER_NODES) -> Net:
    """Zufälliges Reihen-Parallel-Netz mit k Kanten: rekursive Zweiteilung, je Teilung zufällig in Reihe oder parallel.

    Eine Reihenschaltung legt einen neuen inneren Knoten an; ist das Budget `max_inner` verbraucht, wird parallel geteilt (hält die
    Schnittaufzählung klein). Quelle 0, Senke 1 beim Aufbau, danach `normalize`."""
    counter = {"next": 2, "inner": 0}

    def build(kk: int, s: int, t: int) -> list:
        if kk == 1:
            return [(s, t, draw_capacity(rng, capmode))]
        k1 = randint(rng, 1, kk - 1)
        series = rng.uniform() < 0.5 and counter["inner"] < max_inner
        if series:
            mid = counter["next"]
            counter["next"] += 1
            counter["inner"] += 1
            return build(k1, s, mid) + build(kk - k1, mid, t)
        return build(k1, s, t) + build(kk - k1, s, t)

    edges = build(k, 0, 1)
    n = counter["next"]
    # Beim Aufbau ist Knoten 1 die Senke und die inneren Knoten haben die Nummern 2 ... n-1; die Senke wandert auf n-1, die inneren rücken um eins vor.
    remap = {0: 0, 1: n - 1}
    for v in range(2, n):
        remap[v] = v - 1
    return normalize(n, [(remap[a], remap[b], c) for a, b, c in edges], "Reihen-Parallel-Netz")


def bottleneck_chain(rng: SplitMix64, k: int, capmode: str) -> Net:
    """Engpasskette: g parallele Gruppen hintereinander (2 bis 6, höchstens k); die k Kanten werden zufällig auf die Gruppen verteilt, jede hat mindestens eine."""
    g = min(randint(rng, 2, 6), k)
    sizes = [1] * g
    for _ in range(k - g):
        sizes[randint(rng, 0, g - 1)] += 1
    edges = []
    for j, size in enumerate(sizes):
        for _ in range(size):
            edges.append((j, j + 1, draw_capacity(rng, capmode)))
    return Net(g + 1, tuple(edges), "Engpasskette")


def make_net(form: str, k: int, capmode: str, seed: int) -> Net:
    """Das Netz zu den Reglern: ein festes Beispiel (k, Kapazitätsmodus und Zufalls-Seed zählen dann nicht) oder ein Zufallsnetz."""
    if form in C.EXAMPLES:
        return example_net(form)
    rng = SplitMix64(seed)
    if form == "Zufalls-DAG":
        net = random_dag(rng, k, capmode)
    elif form == "Reihen-Parallel-Netz":
        net = series_parallel(rng, k, capmode)
    elif form == "Engpasskette":
        net = bottleneck_chain(rng, k, capmode)
    else:
        raise ValueError(f"unbekannte Netzform: {form}")
    check_net(net)
    return net


def make_random_net_k(k: int, capmode: str, seed: int, n: int | None = None) -> Net:
    """Zufalls-DAG mit k Kanten und n Knoten (für die Größe-Tab und die Messreihen)."""
    return random_dag(SplitMix64(seed), k, capmode, n=n, max_attempts=2000)
