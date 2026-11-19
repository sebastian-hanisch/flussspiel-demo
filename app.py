"""Flussspiel – wem gehört die Strecke - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Betriebe besitzen Strecken eines Netzes; der Wert einer Koalition ist der Fluss, den sie allein durchsetzt. Gezeigt werden Kern (aus Max-Flow-Min-Cut),
Shapley-Wert, Veto- und überzählige Kanten, Ausbau und die Zeilenerzeugung für große Netze. Aufbau nach DEMO-PLAYBOOK §2.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import fsp_constants as C
import fsp_core as K
import fsp_experiments as E
import fsp_game as G
import fsp_net as N
import fsp_scenario as S
import fsp_tables as T
import fsp_texts as X
from fsp_evaluation import (
    Analysis,
    Settings,
    analyse,
    fmt,
    names_of,
    pct,
    verdict_ausbau,
    verdict_e1,
    verdict_e2,
    verdict_e3,
    verdict_hull,
    verdict_ranges,
    verdict_rowgen,
    verdict_shapley_core,
)
from fsp_layout import dunkles_design, kopf_css
from fsp_pdf_export import build_pdf
from fsp_rng import SplitMix64
from fsp_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from fsp_visualization import build_ausbau, build_e1, build_net, build_ranges

st.set_page_config(page_title="Flussspiel – Sebastian Hanisch", layout="wide")


# ------------------------------------------------------------------------------------------------------------------- Cache (Schlüssel = alle Eingaben)
@st.cache_data(show_spinner=False, max_entries=6)
def _analysis(settings: Settings) -> Analysis:
    return analyse(settings)


@st.cache_data(show_spinner=False, max_entries=16)
def _ranges(net: N.Net) -> np.ndarray:
    return K.core_ranges(N.coalition_values(net))


@st.cache_data(show_spinner=False, max_entries=16)
def _hull(net: N.Net, seed: int) -> dict:
    v = N.coalition_values(net)
    return K.hull_gap(net, v, K.directions(net.m, 40, SplitMix64(seed)))


@st.cache_data(show_spinner=False, max_entries=16)
def _ausbau(net: N.Net, edge: int) -> dict:
    return E.monotonicity_edge(net, edge, caps=C.CAP_SWEEP)


@st.cache_data(show_spinner=False)
def _tables() -> dict:
    return T.load()


@st.cache_data(show_spinner=False, max_entries=8)
def _live_e1(seed: int) -> list:
    return [E.shapley_core_cell("Zufalls-DAG", "alle 1", k, C.LIVE_RUNS, seeds=(seed,)) for k in (3, 4, 5, 6, 8)]


def gated(key: str, signature, auto: bool, label: str, compute, note: str = ""):
    """Rechnet sofort (auto) oder erst auf Klick; das Ergebnis bleibt gegen `signature` in st.session_state (klein halten)."""
    held = st.session_state.get(key)
    if auto:
        return compute()
    if held is not None and held[0] == signature:
        return held[1]
    if note:
        st.caption(note)
    if st.button(label, key=f"btn_{key}"):
        t0 = time.perf_counter()
        with st.spinner("Rechne ..."):
            result = compute()
        st.session_state[key] = (signature, result)
        st.caption(f"Rechenzeit {time.perf_counter() - t0:.2f} s (auf diesem Rechner, kein Maßstab für andere).")
        return result
    return None


# ------------------------------------------------------------------------------------------------------------------------------------------ Kopf
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🚰 Flussspiel – wem gehört die Strecke")
st.markdown(
    """
Strecken eines Netzes gehören verschiedenen **Betrieben**. Eine Gruppe von Betrieben kann nur ihre eigenen Strecken nutzen; ihr **Wert** ist der größte Fluss,
den sie so durchsetzt, und jede Einheit bringt allen denselben Erlös (**Modellannahme**). Gefragt ist, wie der Gesamtwert aufgeteilt wird, ohne dass eine Gruppe
mit ihrer eigenen Strecke besser führe: der **Kern**. Er zeigt, wer ein **Engpass** ist (Veto-Kante), wer **überzählig** ist (Redundanz) und was **mehr Kapazität** bringt.
[network-flow-demo](https://sebastianhanisch-network-flow-demo.streamlit.app/) optimiert den Fluss, hier wird er aufgeteilt;
[shapley-demo](https://sebastianhanisch-shapley-demo.streamlit.app/) und [kern-demo](https://sebastianhanisch-kern-demo.streamlit.app/) behandeln dieselben Begriffe am Tourenspiel.

Weiter unten: die Reiter, die Abschnitte **🔬 Experiment**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Flussspiel", expanded=True):
    st.markdown(X.HOW_IT_WORKS)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

# ------------------------------------------------------------------------------------------------------------------------------------ Seitenleiste
with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    form = st.selectbox("Netz", C.FORMS, key="form_select", help="Sechs feste Beispielnetze (von Hand gerechnet) und drei Zufallsnetz-Formen.")
    fixed = form in C.EXAMPLES
    if fixed:
        st.caption("Festes Beispielnetz: Kantenzahl, Kapazitätsbereich und Zufalls-Seed gelten erst bei einem Zufallsnetz.")
    k = st.slider("Zahl der Kanten (Betriebe) k", *bounds("k_slider"), key="k_slider", disabled=fixed,
                  help=f"Nur bei Zufallsnetzen. Alle 2^k Koalitionen werden gerechnet, bis k = {C.K_MAX} (2^{C.K_MAX} Werte).")
    capmode = st.selectbox("Kapazitäten (Zufallsnetze)", list(C.CAPMODES), key="capmode_select", disabled=fixed,
                           help="Alle gleich 1 (Einheitskapazität) oder ganzzahlig zufällig im Bereich.")
    seed = st.number_input("Zufalls-Seed des Netzes", *bounds("seed_input"), key="seed_input", step=1, disabled=fixed)
    st.button("🎲 Neues Zufallsnetz", width="stretch", on_click=randomize_seed, disabled=fixed)

sync_query_params({"form_select": form, "k_slider": int(k), "capmode_select": capmode, "seed_input": int(seed)})

settings = Settings(form=form, k=int(k), capmode=capmode, seed=int(seed)).effective()
if settings.form not in C.EXAMPLES and settings.k > 16:
    with st.spinner(f"Rechne alle {2 ** settings.k:,} Koalitionen ...".replace(",", ".")):
        a = _analysis(settings)
else:
    a = _analysis(settings)
net = a.net
names = N.edge_names(net.m)
sig = (net.n, net.edges)
flow_value, flows = N.max_flow(net)
cut = N.min_cut_edges(net)
assert flow_value == a.v_grand, "Max-Flow und Schnittformel müssen übereinstimmen"   # Ergebnis vor der Anzeige prüfen (DEMO-PLAYBOOK 1d)

# ------------------------------------------------------------------------------------------------------------------------------------- Ergebnis
st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("v(N), Einheiten je Tag", fmt(a.v_grand), help="Wert der großen Koalition: der größte Fluss durch das ganze Netz.")
m2.metric("Betriebe (Kanten)", f"{net.m}")
m3.metric("Veto-Kanten", names_of(a.veto, net.m) or "keine", help="Engpass: ohne diesen Betrieb ist der Wert 0.")
m4.metric("Überzählige Kanten", names_of(a.surplus, net.m) or "keine", help="Redundanz: ohne diesen Betrieb sinkt der Wert nicht.")
st.plotly_chart(build_net(net, flows=flows, cut_edges=cut), width="stretch", key="main_net")
st.caption(f"Farbe = Betrieb, Breite = Kapazität, Beschriftung „Fluss/Kapazität“. Rote Unterlage: Kanten eines minimalen Schnitts "
           f"({names_of(cut, net.m)}), seine Kapazität ist v(N). Netz: {net.n} Knoten, Quelle s links, Senke t rechts.")

kind, verdict_text = verdict_shapley_core(a)
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf"] = (sig, build_pdf(a, verdict_text))
held_pdf = st.session_state.get("pdf")
if held_pdf is not None and held_pdf[0] == sig:
    st.download_button("⬇️ PDF herunterladen", held_pdf[1], file_name="flussspiel.pdf", mime="application/pdf", key="pdf_download")

# ----------------------------------------------------------------------------------------------------------------------------------- Kernfrage
st.markdown("## ❓ Kernfrage: Liegt der Shapley-Wert im Kern, und wem gehört die Strecke?")
getattr(st, kind)(verdict_text)

with st.expander("📖 Benannte Regeln und Verfahren, mit Beispielrechnung"):
    st.markdown(X.TERMS_TABLE)
    st.markdown(
        """
**Shapley-Wert, Beispielspur „Drei Kanten“.** Es gibt 3! = 6 Reihenfolgen der Betriebe A, B, C. Jeder Betrieb bekommt in jeder Reihenfolge, was er dem Wert der schon
versammelten Betriebe hinzufügt, und der Shapley-Wert ist das Mittel. B fügt nur etwas hinzu, wenn A schon da ist und C noch nicht: in der Reihenfolge A, B, C steigt der Wert von 0 auf 3 (v({A, B}) − v({A}) = 3), in den anderen fünf Reihenfolgen ist der Beitrag von B 0. B bekommt daher 3/6 = 1/2. Kern-Zahlungen verlangen dagegen von B 0, weil C allein neben A schon den ganzen Wert 4 durchsetzt.

**Zeilenerzeugung.** Das Kernprogramm hat je Koalition eine Ungleichung, bei k Kanten 2^k − 2 Stück. Statt alle aufzuschreiben, löst man das Programm mit wenigen Zeilen
(am Anfang nur die Einzelbetriebe), sucht dann die am stärksten verletzte Koalition (Trennungsproblem) und fügt sie als neue Zeile hinzu; findet die Suche keine, ist die Lösung im Kern.
        """
    )

st.markdown("---")

# -------------------------------------------------------------------------------------------------------------------------------------- Reiter
tab_net, tab_core, tab_shap, tab_build, tab_big = st.tabs(["🗺️ Netz und Wert", "🧊 Kern", "⚖️ Shapley gegen Kern", "🔧 Ausbau", "📏 Größe"])

# --- Netz und Wert ---------------------------------------------------------------------------------------------------------------------------
with tab_net:
    def _node(x):
        return "s" if x == 0 else ("t" if x == net.sink else str(x))

    rows = []
    for i, (u, w, c) in enumerate(net.edges):
        kind_i = "Veto" if i in a.veto else ("überzählig" if i in a.surplus else ("Nullspieler" if i in a.null else "gewöhnlich"))
        rows.append({"Betrieb": names[i], "Kante": f"{_node(u)} → {_node(w)}", "Kapazität": c, "Fluss in v(N)": flows[i], "Art": kind_i, "im min. Schnitt": "ja" if i in cut else ""})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Veto: ohne diesen Betrieb ist v(N) gleich 0. Überzählig: kein Nullspieler, aber ohne ihn sinkt v(N) nicht. Der Fluss in v(N) ist ein größter Fluss, nicht der einzige.")
    chosen = st.multiselect("Koalition wählen (nur ihre Kanten sind verfügbar)", names, default=names, key=f"coalition_{net.m}")
    mask = sum(1 << i for i, n in enumerate(names) if n in chosen)
    value_s, flows_s = N.max_flow(net, mask)
    c1, c2 = st.columns([1, 2])
    c1.metric(f"v({N.coalition_label(mask, net.m)}), Einheiten je Tag", fmt(float(value_s)))
    c1.caption("Kontrolle: Schnittformel " + ("stimmt" if value_s == a.v[mask] else "WEICHT AB") + " mit dem Fluss überein.")
    c2.plotly_chart(build_net(net, flows=flows_s, highlight=[i for i in range(net.m) if mask >> i & 1]), width="stretch", key="coalition_net")
    if net.m <= 6:
        table = [{"Koalition": N.coalition_label(s, net.m), "v(S)": fmt(float(a.v[s]))} for s in range(1 << net.m)]
        st.markdown("**Alle Koalitionen**")
        st.dataframe(table, hide_index=True, width="stretch", height=min(400, 36 + 35 * len(table)))
    else:
        st.caption(f"Bei {net.m} Kanten gibt es {2 ** net.m:,} Koalitionen; die Tabelle aller Werte gibt es bis 6 Kanten.".replace(",", "."))

# --- Kern ---------------------------------------------------------------------------------------------------------------------------------------
with tab_core:
    st.markdown("**Ecken: Auszahlungen der minimalen Schnitte.** Jede Kante eines kleinsten Schnitts bekommt ihre Kapazität, alle anderen nichts. Diese Vektoren liegen im Kern.")
    corners = K.min_cut_payoffs(net, a.v_grand)
    st.dataframe([{**{names[i]: fmt(float(x[i])) for i in range(net.m)}, "Summe": fmt(float(x.sum()))} for x in corners], hide_index=True, width="stretch")
    ranges = gated(f"ranges_{net.m}", sig, net.m <= C.RANGES_LIVE_MAX, f"▶️ Spielraum je Kante rechnen ({2 * net.m} Programme mit Zeilenerzeugung)",
                   lambda: _ranges(net), note=f"Ab {C.RANGES_LIVE_MAX + 1} Kanten erst auf Klick.")
    if ranges is not None:
        st.markdown("**Spielraum je Kante im Kern** (Balken: kleinste bis größte Zahlung, Raute: Shapley-Wert)")
        st.plotly_chart(build_ranges(net, ranges, a.phi), width="stretch", key="ranges_chart")
        kind_r, text_r = verdict_ranges(a, ranges)
        getattr(st, kind_r)(text_r)
    st.markdown("**Hülle der minimalen Schnitte gegen den Kern**")
    unit = st.toggle("Alle Kapazitäten auf 1 setzen (Einheitskapazität)", key="unit_toggle",
                     help="Bei Einheitskapazität ist der Kern die Hülle der minimalen Schnitte; bei ungleichen Kapazitäten ist er meist größer.")
    net_h = N.Net(net.n, tuple((u, w, 1) for u, w, _ in net.edges), net.name + " (Einheit)") if unit else net
    sig_h = (net_h.n, net_h.edges, int(seed))
    gap = gated(f"hull_{net.m}_{int(unit)}", sig_h, net_h.m <= 10, "▶️ Hülle gegen Kern prüfen (Stichprobe von Richtungen)", lambda: _hull(net_h, int(seed)),
                note="Ab 11 Kanten erst auf Klick.")
    if gap is not None:
        kind_h, text_h = verdict_hull(gap)
        getattr(st, kind_h)(text_h)

# --- Shapley gegen Kern ---------------------------------------------------------------------------------------------------------------------
with tab_shap:
    exact = net.m <= G.EXACT_SHAPLEY_MAX_PLAYERS
    fractions = G.shapley_fractions(a.v) if exact else None
    ranges_s = st.session_state.get(f"ranges_{net.m}", (None, None))[1] if net.m > C.RANGES_LIVE_MAX else _ranges(net)
    table = []
    for i in range(net.m):
        row = {"Betrieb": names[i], "Shapley-Wert": (str(fractions[i]) + f" = {fmt(float(a.phi[i]))}") if exact and fractions[i].denominator != 1 else fmt(float(a.phi[i]))}
        if ranges_s is not None:
            row["Kern von"] = fmt(float(ranges_s[i][0]))
            row["Kern bis"] = fmt(float(ranges_s[i][1]))
        row["Art"] = "Veto" if i in a.veto else ("überzählig" if i in a.surplus else ("Nullspieler" if i in a.null else "gewöhnlich"))
        table.append(row)
    st.dataframe(table, hide_index=True, width="stretch")
    if ranges_s is None:
        st.caption("Die Kern-Spalten erscheinen, sobald im Reiter „Kern“ der Spielraum gerechnet ist.")
    st.metric("Kleinster Überschuss x(S) − v(S) des Shapley-Werts über alle echten Koalitionen", fmt(a.margin))
    if not a.in_core and a.v_grand > 0:
        sums = G.coalition_sums(a.phi) - a.v
        sums[0] = sums[-1] = 0.0
        worst = np.argsort(sums)[: min(5, len(sums))]
        st.markdown("**Am stärksten verletzte Koalitionen**")
        st.dataframe([{"Koalition": N.coalition_label(int(s), net.m), "Shapley-Summe x(S)": fmt(float(G.coalition_sums(a.phi)[s])), "Wert v(S)": fmt(float(a.v[s])),
                       "Überschuss": fmt(float(sums[s]))} for s in worst if sums[s] < -1e-9], hide_index=True, width="stretch")
    if a.surplus and a.v_grand > 0:
        share = float(a.phi[list(a.surplus)].sum() / a.v_grand)
        st.info(f"Auf die überzählige(n) Kante(n) {names_of(a.surplus, net.m)} entfallen {pct(share)} des Wertes im Shapley-Wert; im Kern bekommen sie 0.")
    with st.expander("Handbeweis: überzählige Kante ⇒ Shapley-Wert außerhalb des Kerns"):
        st.markdown(X.BEWEIS_H4)
    st.caption("Der Restfall (keine überzählige Kante, Shapley-Wert trotzdem außerhalb) steht im Preset „Vier Kanten“: dort sind alle Grenzbeiträge positiv.")

# --- Ausbau -------------------------------------------------------------------------------------------------------------------------------------
with tab_build:
    if net.m > C.AUSBAU_MAX:
        st.info(f"Der Ausbau-Reiter rechnet bis {C.AUSBAU_MAX} Kanten (je Kapazitätsschritt zwei Programme); das Netz hat {net.m}.")
    else:
        edge_name = st.selectbox("Welche Kante wird ausgebaut?", names, key=f"ausbau_kante_{net.m}")
        edge = names.index(edge_name)
        res = _ausbau(net, edge)
        st.plotly_chart(build_ausbau(res, edge_name), width="stretch", key="ausbau_chart")
        kind_a, text_a = verdict_ausbau(res, edge_name)
        getattr(st, kind_a)(text_a)
        st.caption(f"Nur die Kapazität von Kante {edge_name} läuft von {C.CAP_SWEEP[0]} bis {C.CAP_SWEEP[-1]}, alle anderen bleiben.")
        st.markdown("**Eigene Kapazität eingeben**")
        cap_now = net.edges[edge][2]
        new_cap = st.number_input(f"Kapazität der Kante {edge_name} (aktuell {cap_now}; 0 streicht die Kante)", 0, C.CAP_MAX, cap_now, key=f"ausbau_cap_{net.m}_{edge}")
        net2 = net.with_capacity(edge, int(new_cap))
        v2 = N.coalition_values(net2)
        phi2 = G.shapley(v2)
        lo2, hi2 = K.core_range(v2, edge)
        c1, c2, c3 = st.columns(3)
        c1.metric("v(N)", fmt(float(v2[-1])), delta=fmt(float(v2[-1] - a.v_grand)) if v2[-1] != a.v_grand else None)
        c2.metric(f"Shapley-Wert von {edge_name}", fmt(float(phi2[edge])), delta=fmt(float(phi2[edge] - a.phi[edge])) if abs(phi2[edge] - a.phi[edge]) > 1e-12 else None)
        c3.metric(f"Kern von {edge_name}", f"{fmt(lo2)} bis {fmt(hi2)}")

# --- Größe -------------------------------------------------------------------------------------------------------------------------------------
with tab_big:
    st.markdown(f"Der Kern wird für große Zufallsnetze nie über alle Koalitionen aufgeschrieben: die Zeilenerzeugung braucht wenige Dutzend Zeilen statt bis zu 2^{C.K_BIG_MAX}. "
                "Bis 20 Kanten sucht die Aufzählung die verletzte Koalition, darüber ein gemischt-ganzzahliges Programm (Zeitgrenze 20 s je Trennung).")
    big_k = st.slider("Zahl der Kanten", C.K_BIG_MIN, C.K_BIG_MAX, C.DEFAULT_K_BIG, key="big_k")
    big_seed = st.number_input("Zufalls-Seed", 0, C.SEED_MAX, 1, key="big_seed", step=1)
    big_sig = (int(big_k), int(big_seed))

    def _run_big():
        net_b = S.make_random_net_k(int(big_k), "1 bis 5", int(big_seed), n=C.BIG_NODES if big_k > 20 else None)
        cut_b = N.min_cut_edges(net_b)
        edge_b = cut_b[S.randint(SplitMix64(int(big_seed) + 7), 0, len(cut_b) - 1)]
        return net_b, edge_b, K.max_payment_by_row_generation(net_b, edge_b, "auto", time_limit=20.0)

    out = gated("big", big_sig, False, "▶️ Größte Zahlung einer Kante im Kern rechnen", _run_big,
                note="Die Kante wird zufällig aus einem minimalen Schnitt gewählt (ihr Kern-Maximum ist dann größer als 0).")
    if out is not None:
        net_b, edge_b, res_b = out
        method = "Aufzählung" if net_b.m <= K.SEPARATION_ENUM_MAX else "gemischt-ganzzahliges Programm (HiGHS)"
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Kanten / Knoten", f"{net_b.m} / {net_b.n}")
        b2.metric("Koalitionen insgesamt", f"{2 ** net_b.m:,}".replace(",", "."))
        b3.metric("Zeilen im Programm", f"{res_b.rows}")
        b4.metric("Runden", f"{res_b.rounds}")
        kind_b, text_b = verdict_rowgen(res_b, net_b.m, edge_b)
        getattr(st, kind_b)(text_b)
        st.caption(f"Trennung per {method}; Status des Lösers: {res_b.status}; Trennung {res_b.separation_seconds:.2f} s von {res_b.total_seconds:.2f} s insgesamt.")

st.markdown("---")

# ------------------------------------------------------------------------------------------------------------------------------- Experimente
data = _tables()
if not T.is_current(data):
    st.warning("Die vorgerechneten Messreihen (data/messreihen.json) stammen von einem älteren Stand des Codes: `python tools/make_messreihen.py` neu laufen lassen.")
cells = data["e1"]


def _series(form_, cap_):
    return sorted((c for c in cells if c["form"] == form_ and c["capmode"] == cap_), key=lambda c: c["k"])


st.subheader("🔬 Experiment 1: Wie oft liegt der Shapley-Wert im Kern, wenn das Netz wächst?")
st.caption(f"Vorgerechnet: Anteil der Zufallsnetze mit Shapley-Wert im Kern, Mittel über {len(data['meta']['sweep_seeds'])} Zufalls-Seeds ({data['meta']['e1_runs']} Netze je Seed und Zelle, "
           f"ab k = 16: {data['meta']['e1_big_runs']}). Nur Kanten an s-t-Wegen zählen als Betriebe.")
series = {f"{f}, {c}": _series(f, c) for f, c in (("Zufalls-DAG", "alle 1"), ("Zufalls-DAG", "1 bis 5"), ("Reihen-Parallel-Netz", "alle 1"), ("Engpasskette", "alle 1"))}
st.plotly_chart(build_e1(series), width="stretch", key="e1_chart")
kind_e1, text_e1 = verdict_e1(series["Zufalls-DAG, alle 1"])
getattr(st, kind_e1)(text_e1)
ks_all = sorted({c["k"] for c in cells})
table_e1 = []
for f, c in (("Zufalls-DAG", "alle 1"), ("Zufalls-DAG", "1 bis 5"), ("Reihen-Parallel-Netz", "alle 1"), ("Reihen-Parallel-Netz", "1 bis 5"),
             ("Engpasskette", "alle 1"), ("Engpasskette", "1 bis 5")):
    row = {"Netzform, Kapazität": f"{f}, {c}"}
    by_k = {x["k"]: x for x in _series(f, c)}
    for kk in ks_all:
        row[f"k = {kk}"] = f"{100 * by_k[kk]['in_core_mean']:.0f} %" if kk in by_k else "–"
    table_e1.append(row)
st.dataframe(table_e1, hide_index=True, width="stretch")
dag_unit = _series("Zufalls-DAG", "alle 1")
st.caption("Überzählige Kante und Shapley im Kern kamen in keiner Zelle zugleich vor (Zählung über alle Zellen der Tabelle). "
           f"Anteil der Netze mit überzähliger Kante im Zufalls-DAG bei Kapazität 1: {pct(dag_unit[0]['surplus_share'])} bei k = {dag_unit[0]['k']}, "
           f"{pct(dag_unit[-1]['surplus_share'])} bei k = {dag_unit[-1]['k']}.")
live_seed = st.number_input("Zufalls-Seed für den Kurzlauf", 0, C.SEED_MAX, 1, key="e1_seed", step=1)
if st.button(f"▶️ Kurzlauf: Zufalls-DAG, Kapazität 1, {C.LIVE_RUNS} Netze je k", key="e1_live"):
    t0 = time.perf_counter()
    with st.spinner("Rechne 5 Zellen ..."):
        live = _live_e1(int(live_seed))
    st.dataframe([{"k": c["k"], "Netze": c["runs"], "Shapley im Kern": pct(c["in_core_mean"]), "mit überzähliger Kante": pct(c["surplus_share"])} for c in live], hide_index=True)
    st.caption(f"Rechenzeit {time.perf_counter() - t0:.1f} s. {C.LIVE_RUNS} Netze je Zelle schwanken stark; die Tabelle oben hat {data['meta']['e1_runs'] * len(data['meta']['sweep_seeds'])}.")

st.markdown("---")
st.subheader("🔬 Experiment 2: Ist der Kern die Hülle der minimalen Schnitte?")
hull_rows = data["e2"]["hull"]
kind_e2, text_e2 = verdict_e2(hull_rows)
getattr(st, kind_e2)(text_e2)
st.dataframe([{"Kapazitäten": r["capmode"], "Netze": r["runs"], "Kern = Hülle in allen Richtungen": pct(r["equal_mean"]),
               "Lücke bei Abweichung (Median, Anteil von v(N))": pct(r["gap_median"]) if r["gap_median"] else "–", "Richtungen je Netz (bis zu)": r["directions"]} for r in hull_rows],
             hide_index=True, width="stretch")
st.caption("Zufalls-DAG, k = 4 bis 8; je Netz 40 Zufallsrichtungen und die 2k Koordinatenrichtungen. Spielraum je Kante nach Klasse (Kern-Obergrenze, in Anteilen von v(N), Mittel über die Kanten der Klasse):")
class_rows = []
for r in data["e2"]["classes"]:
    for key, label in (("veto", "Veto"), ("surplus", "überzählig"), ("other", "übrige")):
        c = r[key]
        class_rows.append({"Kapazitäten": r["capmode"], "Klasse": label, "Kanten": c["edges"], "Kern-Obergrenze": fmt(c["max_mean"]) if c["max_mean"] is not None else "–",
                           "davon mit Obergrenze 0": c["max_is_zero"]})
st.dataframe(class_rows, hide_index=True, width="stretch")
st.caption("Live für das Netz oben: Reiter „Kern“, Umschalter „Einheitskapazität“.")

st.markdown("---")
st.subheader("🔬 Experiment 3: Steigt die Zahlung, wenn die eigene Kapazität steigt?")
e3 = data["e3"]
kind_e3, text_e3 = verdict_e3(e3)
getattr(st, kind_e3)(text_e3)
st.caption("Kapazität einer Kante von 1 bis 8, die übrigen Kanten fest; Zufalls-DAG, Kapazitäten 1 bis 5. Live für jede Kante des Netzes oben: Reiter „Ausbau“.")

st.markdown("---")
st.subheader("🔬 Experiment 4: Wie viele Zeilen braucht der Kern bei großen Netzen?")
e4 = data["e4"]
st.dataframe([{"Kanten": r["m"], "Trennung": "Aufzählung" if r["method"] == "enum" else "gemischt-ganzzahlig", "Netze": r["runs"], "Koalitionen": f"{r['coalitions']:,}".replace(",", "."),
               "Zeilen (Mittel)": f"{r['rows_mean']:.1f}".replace(".", ","), "Zeilen (kleinste bis größte)": f"{r['rows_min']} bis {r['rows_max']}",
               "alle optimal": "ja" if r["all_optimal"] else "nein", "Zielwert gleich volles Programm": f"{r['equal_to_full']} von {r['runs']}" if r["m"] <= 14 else "nicht gerechnet"} for r in e4],
             hide_index=True, width="stretch")
last4 = e4[-1]
if last4["all_optimal"]:
    st.success(f"Bei {last4['m']} Kanten ({last4['coalitions']:.1e} Koalitionen) brauchte die Zeilenerzeugung {last4['rows_min']} bis {last4['rows_max']} Zeilen "
               f"(über {last4['runs']} Netze, in allen bewiesen optimal).")
else:
    st.warning(f"Bei {last4['m']} Kanten wurde nicht in allen {last4['runs']} Netzen der Optimalitätsbeweis erreicht (Zeitgrenze oder Löserfehler der Trennung).")
st.caption("Zielgröße: größte Zahlung einer zufällig gewählten Kante eines minimalen Schnitts. Live: Reiter „Größe“.")

st.markdown("---")
st.subheader("🔬 Experiment 5: Wann liegt der Shapley-Wert außerhalb, obwohl keine Kante überzählig ist?")
rf = data["restfall"]
st.info(f"In {rf['nets']} Zufalls-DAG (k = {rf['ks'][0]} bis {rf['ks'][-1]}, Kapazitäten 1 bis 5) waren {rf['convex']} Spiele konvex; in allen lag der Shapley-Wert im Kern "
        f"({rf['convex_in_core']} von {rf['convex']}). Ohne überzählige Kante gab es {rf['clean']} Netze, in {rf['clean_outside']} davon lag der Shapley-Wert außerhalb; "
        f"alle {rf['clean_outside_nonconvex']} dieser Spiele waren nicht konvex. Überzählige Kante und Shapley im Kern zugleich: {rf['surplus_and_in_core']}-mal.")
st.caption("Konvex heißt: der Grenzbeitrag eines Betriebs wächst mit der Koalition. Dann liegt der Shapley-Wert immer im Kern (Standardsatz); das Beispiel für den Restfall ist das Preset „Vier Kanten“.")

st.markdown("---")

# ------------------------------------------------------------------------------------------------------------------------------------ Grenzen
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein Eigentümer je Kante** | Gehört einem Betrieb mehr als eine Kante, ist er ein Spieler mit mehreren Strecken; das Spiel und seine Spielerzahl ändern sich. Nicht gerechnet. | niemand in dieser Demo |
| **Gleicher Erlös je Einheit, ein Gut** | Unterschiedliche Erlöse oder mehrere Güter machen aus dem Fluss ein Mehrgüterproblem. Nicht gerechnet. | niemand in dieser Demo |
| **Gerichtete Kanten ohne Kosten** | Kosten je Einheit machen aus dem Wert eine Differenz aus Erlös und Kosten; die Schnittformel gilt dann nicht mehr. Nicht gerechnet. | [network-flow-demo](https://sebastianhanisch-network-flow-demo.streamlit.app/) rechnet kostenminimale Flüsse, aber ohne Aufteilung |
| **Jede Koalition nutzt ihre Kanten allein** | Müssen Strecken gemeinsam mit anderen genutzt werden (Netzzugang), ist der Wert einer Koalition kleiner als ihr Fluss. Nicht gerechnet. | niemand in dieser Demo |
| **Nur Kanten an s-t-Wegen zählen als Betriebe** | Eine Kante außerhalb jedes Weges ist ein Nullspieler: sie bekommt in Shapley-Wert und Kern 0 und verändert keinen Wert, würde aber jede Zählung über Betriebe verwässern. | die Zufallsnetze der Demo schließen sie aus |
| **Netze ohne Kreise** | Das Netzbild und die Erzeuger gehen von Kanten in einer Richtung aus; die Schnittformel selbst gilt auch bei Kreisen, geprüft ist es hier nicht. | niemand in dieser Demo |
"""
)

st.markdown("---")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(X.ASSUMPTIONS_MARKDOWN)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist ein Netz mit Kanten, jede mit Besitzer und Kapazität. Gesucht ist für jede Gruppe von Besitzern der größte Fluss, den sie allein durchsetzt, und
eine Aufteilung des Gesamtwerts, die keine Gruppe schlechterstellt als ihr eigener Fluss.

| Symbol | Bedeutung |
|---|---|
| $V$, $s$, $t$ | Knoten, Quelle, Senke |
| $E$ | Kanten; jede Kante $e$ ist ein Spieler |
| $c_e$ | Kapazität der Kante $e$ (ganzzahlig, mindestens 0) |
| $S \subseteq E$ | Koalition: die Kanten, die ihre Mitglieder zur Verfügung stellen |
| $f_e$ | Fluss auf der Kante $e$ |
| $\delta^+(u)$, $\delta^-(u)$ | Kanten, die aus $u$ hinaus- bzw. in $u$ hineinführen |
| $\delta^+(A)$ | Kanten, die aus der Knotenmenge $A$ hinausführen |
| $v(S)$ | Wert der Koalition $S$ (größter Fluss über ihre Kanten) |
| $x(S)$ | Summe $\sum_{e \in S} x_e$ der Zahlungen an $S$ |
| $\varphi_i$ | Shapley-Wert des Spielers $i$ |

**Wert einer Koalition** (lineares Programm; Variablen $f_e$, alle für $e \notin S$ gleich 0):
        """
    )
    st.markdown(r"$$v(S) = \max \; \sum_{e \in \delta^+(s)} f_e - \sum_{e \in \delta^-(s)} f_e$$")
    st.markdown(r"$$\sum_{e \in \delta^-(u)} f_e = \sum_{e \in \delta^+(u)} f_e \qquad \text{für alle } u \in V \setminus \{s, t\}$$")
    st.markdown(r"$$0 \le f_e \le c_e \quad (e \in S), \qquad f_e = 0 \quad (e \notin S)$$")
    st.markdown("**Max-Flow-Min-Cut** (Schnittformel, so rechnet die App alle $2^k$ Werte):")
    st.markdown(r"$$v(S) = \min_{A \subseteq V,\; s \in A,\; t \notin A} \; \sum_{e \in \delta^+(A) \cap S} c_e$$")
    st.markdown("**Kern** (Gewinnspiel: keine Gruppe bekommt weniger als sie allein durchsetzt):")
    st.markdown(r"$$\mathrm{Kern} = \{ x \in \mathbb{R}^E : \; x(E) = v(E), \;\; x(S) \ge v(S) \;\; \text{für alle } S \subseteq E \}$$")
    st.markdown("**Schnitt-Auszahlung** eines kleinsten Schnitts $A$: Die Kanten des Schnitts bekommen ihre Kapazität, alle anderen nichts. Sie liegt im Kern.")
    st.markdown(r"$$x^A_e = c_e \;\; \text{für } e \in \delta^+(A), \qquad x^A_e = 0 \;\; \text{sonst}$$")
    st.markdown("**Shapley-Wert** (Mittel des Grenzbeitrags über alle Reihenfolgen):")
    st.markdown(r"$$\varphi_i = \sum_{S \subseteq E \setminus \{i\}} \frac{|S|! \, (|E| - |S| - 1)!}{|E|!} \, \bigl( v(S \cup \{i\}) - v(S) \bigr)$$")
    st.markdown("**Überzählige Kante** $i$ (Grenzbeitrag 0 zur großen Koalition): im Kern gilt $x_i = 0$, der Shapley-Wert ist positiv, wenn $i$ in einer Reihenfolge einen positiven Beitrag leistet.")
    st.markdown(r"$$v(E \setminus \{i\}) = v(E) \;\; \Longrightarrow \;\; x_i = 0 \;\; \text{für alle } x \in \mathrm{Kern}$$")
    st.markdown("**Zeilenerzeugung:** Das Hauptprogramm maximiert die Zahlung an eine Kante über die Zeilen, die bisher bekannt sind; das Trennungsproblem sucht die am stärksten verletzte Koalition:")
    st.markdown(r"$$\max_{S \subseteq E} \; v(S) - x(S)$$")
    st.markdown(r"Als gemischt-ganzzahliges Programm mit $y_e \in \{0, 1\}$ (Kante gehört zu $S$) und den Flussbedingungen von oben, aber $f_e \le c_e \, y_e$:")
    st.markdown(r"$$\max_{f,\, y} \;\; \sum_{e \in \delta^+(s)} f_e - \sum_{e \in \delta^-(s)} f_e - \sum_{e \in E} x_e \, y_e$$")
    st.markdown(r"$$f_e \le c_e \, y_e, \qquad y_e \in \{0, 1\}, \qquad 0 \le f_e \le c_e$$")
    st.markdown("Ist der Zielwert höchstens 0, liegt $x$ im Kern. Die Formeln sind das vollständige Modell der App: jede Größe ist definiert, Nebenbedingungen und Variablenbereiche stehen dabei.")
    st.markdown(X.LITERATUR_HEAD)
    st.markdown("\n".join(X.LITERATUR))

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Koalitionen und Teilen: Macht und Aufteilung](https://sebastianhanisch.net/konzepte-koalitionen-teilen.html)."
)
