"""Kennzahlen und Verdikte. Jedes Verdikt ist eine reine Funktion `verdict_*(...) -> (Art, Text)`: die App ruft `getattr(st, Art)(Text)`,
`tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an handgesetzten Werten.

Das Spiel ist ein Gewinnspiel (Erlös je Einheit): Kern heißt x(S) >= v(S). Eigennutz und Erlös je Einheit sind Modellannahmen."""

from dataclasses import dataclass

import numpy as np

import fsp_constants as C
import fsp_game as G
import fsp_net as N
import fsp_scenario as S


@dataclass(frozen=True)
class Settings:
    form: str = C.DEFAULT_FORM
    k: int = C.DEFAULT_K
    capmode: str = C.DEFAULT_CAPMODE
    seed: int = C.DEFAULT_SEED

    def effective(self) -> "Settings":
        """Bei einem festen Beispielnetz zählen k, Kapazitätsmodus und Zufalls-Seed nicht: sie werden auf die Standardwerte gesetzt (gleicher Cache-Schlüssel)."""
        if self.form in C.EXAMPLES:
            return Settings(form=self.form)
        return self


@dataclass(frozen=True)
class Analysis:
    net: N.Net
    v: np.ndarray            # Koalitionswerte, Länge 2^m
    phi: np.ndarray          # Shapley-Wert
    margin: float            # kleinster Überschuss x(S) - v(S) des Shapley-Werts über alle echten Koalitionen
    worst: int               # Maske der am stärksten verletzten (bzw. knappsten) Koalition
    surplus: tuple           # überzählige Kanten (Indizes)
    veto: tuple
    null: tuple              # Nullspieler (Indizes)
    convex: object           # True/False, None bei mehr als C.CONVEX_CHECK_MAX Kanten

    @property
    def m(self) -> int:
        return self.net.m

    @property
    def v_grand(self) -> float:
        return float(self.v[-1])

    @property
    def in_core(self) -> bool:
        return self.margin >= -1e-9


def net_of(settings: Settings) -> N.Net:
    return S.make_net(settings.form, settings.k, settings.capmode, settings.seed)


def analyse_net(net: N.Net) -> Analysis:
    v = N.coalition_values(net)
    phi = G.shapley(v)
    margin, _gap, worst = G.core_margin(v, phi)
    null = G.null_players(v)
    return Analysis(net, v, phi, margin, worst, tuple(G.surplus_edges(v)), tuple(G.veto_edges(v)), tuple(i for i, z in enumerate(null) if z),
                    G.is_convex(v) if net.m <= C.CONVEX_CHECK_MAX else None)


def analyse(settings: Settings) -> Analysis:
    return analyse_net(net_of(settings))


def names_of(indices, m: int) -> str:
    names = N.edge_names(m)
    return ", ".join(names[i] for i in indices)


def fmt(x: float) -> str:
    """Zahl mit Dezimalkomma: ganze Zahlen ohne Nachkomma, sonst drei Stellen (angezeigt wird gerundet, gerechnet nicht)."""
    if abs(x - round(x)) < 5e-10:
        return str(int(round(x)))
    return f"{x:.3f}".replace(".", ",")


def pct(x: float) -> str:
    return f"{100 * x:.1f}".replace(".", ",") + " %"


# ---------------------------------------------------------------------------------------------------------------------- Kernfrage
def verdict_shapley_core(a: Analysis) -> tuple:
    """Kernfrage: Liegt der Shapley-Wert im Kern? Fünf Fälle: kein Wert, ein Spieler, im Kern, außerhalb mit überzähliger Kante, außerhalb ohne (Restfall)."""
    if a.v_grand <= 0:
        return "warning", "Durch dieses Netz fließt nichts (v(N) = 0): es gibt nichts zu verteilen, Shapley-Wert und Kern sind 0."
    if a.m == 1:
        return "info", f"Nur ein Betrieb: ihm gehört der ganze Wert {fmt(a.v_grand)}; Shapley-Wert und Kern fallen zusammen."
    if a.in_core and a.surplus:
        return "error", "Widerspruch zum Satz „überzählige Kante ⇒ Shapley-Wert außerhalb des Kerns“: bitte melden."
    if a.in_core:
        return "success", (f"Der Shapley-Wert liegt **im Kern**: keine Gruppe von Betrieben kann sich mit ihrer eigenen Strecke besser stellen "
                           f"(knappste Koalition: Überschuss {fmt(a.margin)}).")
    phi_surplus = float(a.phi[list(a.surplus)].sum()) if a.surplus else 0.0
    violated = f"Koalition {N.coalition_label(a.worst, a.m)} bekäme {fmt(float(np.sum([a.phi[i] for i in range(a.m) if a.worst >> i & 1])))} statt {fmt(float(a.v[a.worst]))}"
    if a.surplus:
        who = names_of(a.surplus, a.m)
        plural = "haben" if len(a.surplus) > 1 else "hat"
        return "warning", (f"Der Shapley-Wert liegt **außerhalb** des Kerns. Kante {who} {plural} Grenzbeitrag 0 zur großen Koalition und bekommt im Kern 0, "
                           f"im Shapley-Wert zusammen {fmt(phi_surplus)} von {fmt(a.v_grand)}. Verletzt: {violated}.")
    return "warning", (f"Der Shapley-Wert liegt **außerhalb** des Kerns, obwohl jede Kante gebraucht wird (kein Grenzbeitrag 0): Restfall. "
                       f"{'Das Spiel ist nicht konvex. ' if a.convex is False else ''}Verletzt: {violated}.")


# ----------------------------------------------------------------------------------------------------------------- Kern-Tab (Hülle)
def verdict_hull(gap: dict) -> tuple:
    """Kern gegen die Hülle der minimalen Schnitte im Netz der App; `gap` kommt aus `fsp_core.hull_gap`."""
    n = gap["directions"]
    if gap["equal"]:
        return "success", (f"In allen {n} getesteten Richtungen reicht die Hülle der {gap['n_cuts']} minimalen Schnitte an den Kern heran "
                           f"(eine Stichprobe von Richtungen, kein Beweis für alle).")
    return "info", (f"Der Kern ist **größer** als die Hülle der {gap['n_cuts']} minimalen Schnitte: in {gap['with_gap']} von {n} Richtungen "
                    f"liegt der Kern darüber, im größten Fall um {pct(gap['max_gap_rel'])} von v(N).")


def verdict_ranges(a: Analysis, ranges) -> tuple:
    """Spielraum je Kante im Kern: wer kann alles bekommen, wer nichts?"""
    if a.v_grand <= 0:
        return "warning", "Kein Fluss: der Kern besteht nur aus dem Nullvektor."
    zero = [i for i in range(a.m) if ranges[i][1] < 1e-7]
    full = [i for i in range(a.m) if ranges[i][1] > a.v_grand - 1e-7]
    parts = []
    if full:
        parts.append(f"Kante {names_of(full, a.m)} kann im Kern den ganzen Wert {fmt(a.v_grand)} beanspruchen")
    if zero:
        parts.append(f"Kante {names_of(zero, a.m)} bekommt im Kern höchstens 0")
    if not parts:
        return "info", "Keine Kante kann den ganzen Wert beanspruchen, und keine bekommt im Kern höchstens 0; die Obergrenzen liegen zwischen 0 und v(N), jeweils ausschließlich."
    return "info", "; ".join(parts) + "."


# ------------------------------------------------------------------------------------------------------------------- Ausbau (E3)
def verdict_ausbau(result: dict, edge_name: str) -> tuple:
    """Kapazität einer Kante wächst: Shapley-Wert und Kern-Untergrenze der Kante (aus `fsp_experiments.monotonicity_edge`)."""
    caps = result["caps"]
    if result["core_min_falls"] and not result["shapley_falls"]:
        i = next(j for j in range(1, len(caps)) if result["core_min"][j] < result["core_min"][j - 1] - 1e-7)
        return "warning", (f"Gegenläufig: der Shapley-Wert von Kante {edge_name} fällt nie, aber ihre Kern-Untergrenze sinkt von "
                           f"{fmt(result['core_min'][i - 1])} (Kapazität {caps[i - 1]}) auf {fmt(result['core_min'][i])} (Kapazität {caps[i]}).")
    if result["shapley_falls"]:
        return "error", "Der Shapley-Wert sinkt bei steigender Kapazität: das widerspricht der Messung, bitte melden."
    return "info", f"Bei Kante {edge_name} steigen Shapley-Wert und Kern-Untergrenze mit der Kapazität nicht gegenläufig (Kapazität {caps[0]} bis {caps[-1]})."


# ---------------------------------------------------------------------------------------------------------------- Messreihen (E1 bis E4)
def verdict_e1(cells: list) -> tuple:
    """E1 aus Zellen {k, in_core_mean, runs}: erste gegen letzte Kantenzahl; die Zählung steht im Text."""
    cells = sorted(cells, key=lambda c: c["k"])
    first, last = cells[0], cells[-1]
    zero_from = next((c["k"] for c in cells if all(d["in_core_mean"] == 0 for d in cells if d["k"] >= c["k"])), None)
    text = (f"Bei k = {first['k']} Kanten liegt der Shapley-Wert in {pct(first['in_core_mean'])} der Netze im Kern, bei k = {last['k']} in {pct(last['in_core_mean'])} "
            f"({first['runs']} bzw. {last['runs']} Netze).")
    if zero_from is not None and zero_from > first["k"]:
        text += f" Ab k = {zero_from} gab es in dieser Messreihe keinen Treffer."
    return ("success" if last["in_core_mean"] < first["in_core_mean"] else "info"), text


def verdict_e2(hull_rows: list) -> tuple:
    """E2 aus Zeilen {capmode, equal_mean, runs}: Anteil der Netze, in denen Kern und Hülle in allen getesteten Richtungen übereinstimmen."""
    by = {r["capmode"]: r for r in hull_rows}
    unit, other = by["alle 1"], [r for r in hull_rows if r["capmode"] != "alle 1"]
    worst = ", ".join(f"{r['capmode']}: {pct(r['equal_mean'])}" for r in other)
    return "info", (f"Bei Kapazität 1 stimmen Kern und Hülle in {pct(unit['equal_mean'])} der {unit['runs']} Netze überein (in den getesteten Richtungen); "
                    f"bei ungleichen Kapazitäten nur in {worst}.")


def verdict_e3(table: dict) -> tuple:
    """E3: in wie vielen Kante-Netz-Paaren fällt der Shapley-Wert, die Kern-Untergrenze, die Kern-Obergrenze bei steigender eigener Kapazität?"""
    return "info", (f"In {table['pairs']} Kante-Netz-Paaren (k = {', '.join(map(str, table['ks']))}) fällt der Shapley-Wert der Kante bei steigender Kapazität "
                    f"{table['shapley_falls']}-mal, die Kern-Untergrenze {table['core_min_falls']}-mal, die Kern-Obergrenze {table['core_max_falls']}-mal.")


def verdict_rowgen(res, m: int, i: int) -> tuple:
    """E4: Ergebnis der Zeilenerzeugung (RowGenResult) für Kante i mit dem Status des Lösers; „optimal“ nur, wenn die Trennung bis zum Ende bewiesen hat."""
    name = N.edge_names(m)[i]
    coalitions = f"{(1 << m) - 2:,}".replace(",", ".")
    if res.status == "optimal":
        return "success", (f"Größte Zahlung an Kante {name} im Kern: {fmt(float(res.x[i]))}. "
                           f"Das Programm brauchte {res.rows} Zeilen statt {coalitions} und {res.rounds} Runden.")
    if res.status == "Zeitgrenze der Trennung":
        return "warning", f"Zeitgrenze der Trennung nach {res.rounds} Runden und {res.rows} Zeilen: das Ergebnis ist **nicht** als optimal bewiesen."
    if res.status == "Löserfehler der Trennung":
        return "error", (f"Löserfehler der Trennung nach {res.rounds} Runden und {res.rows} Zeilen (das gemischt-ganzzahlige Programm endete weder optimal noch an der Zeitgrenze): "
                         "das Ergebnis ist **nicht** als optimal bewiesen.")
    return "error", f"Zeilenerzeugung ohne Ergebnis (Status: {res.status})."
