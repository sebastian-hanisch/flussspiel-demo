"""PDF-Export (fpdf2, Kernschriften): Netz, Kennzahlen, Shapley-Wert und das Verdikt der Kernfrage. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from fpdf import FPDF

import fsp_evaluation as V
import fsp_net as N

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "≠": "!=", "−": "-", "„": '"', "“": '"', "”": '"', "→": "->", "·": "*", "…": "...", "’": "'", "‘": "'", "Σ": "Summe",
            "∈": "in", "⇒": "=>", "∅": "{}", "✓": "ok"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1 (Umlaute gehören dazu): Sonderzeichen ersetzen, Markdown-Zeichen entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: V.Analysis, verdict_text: str) -> bytes:
    net = a.net
    names = N.edge_names(net.m)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Flussspiel - wem gehört die Strecke"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Netz: {net.name or 'eigenes Netz'}, {net.n} Knoten, {net.m} Kanten (Betriebe). Wert der großen Koalition v(N) = {V.fmt(a.v_grand)} Einheiten je Tag."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Kanten, Kapazität und Shapley-Wert", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for i, (u, w, c) in enumerate(net.edges):
        kind = "Veto" if i in a.veto else ("überzählig" if i in a.surplus else ("Nullspieler" if i in a.null else ""))
        node = lambda x: "s" if x == 0 else ("t" if x == net.sink else str(x))  # noqa: E731
        pdf.multi_cell(0, 5, _latin(f"Betrieb {names[i]}: Kante {node(u)} -> {node(w)}, Kapazität {c}, Shapley-Wert {V.fmt(float(a.phi[i]))}{', ' + kind if kind else ''}"),
                       new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Kernfrage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(verdict_text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Erlös je Einheit, Eigentum je Kante und Kapazitäten sind Modellannahmen, keine Aussagen über reale Betriebe oder Netze. "
                                "Gewinnspiel: im Kern gilt x(S) >= v(S) für jede Koalition S."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
