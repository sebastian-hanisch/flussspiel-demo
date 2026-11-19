"""Feste Texte der App, die auch in der README stehen (ein Test vergleicht sie): Annahmen, Grenzen, bekannte Probleme.

Zahlen stehen hier nur, wenn sie Einstellungen sind (Regler, Raster) oder Handrechnungen der Beispielnetze (in tests/test_algorithm.py geprüft);
Messzahlen kommen mit f-String aus der Rechnung oder stehen in der README mit Testverweis."""

import fsp_constants as C

ASSUMPTIONS_MARKDOWN = f"""\
**Annahmen**

- Ein gerichtetes Netz ohne Kreise mit Quelle und Senke. Jede Kante gehört genau einem Betrieb und hat eine ganzzahlige Kapazität (Einheiten je Tag). Ein Betrieb ist ein Spieler, ein Spieler besitzt genau eine Kante.
- Jede durchgesetzte Einheit bringt allen denselben Erlös; der Wert einer Koalition ist der größte Fluss, den ihre Kanten allein durchsetzen. Das ist eine Modellannahme, kein Befund über reale Netze.
- Der Wert ist ein Gewinn (Erlös), kein Kostenspiel: im Kern gilt x(S) ≥ v(S), keine Gruppe bekommt weniger als sie allein durchsetzt.
- Zufallsnetze enthalten nur Kanten an Wegen von der Quelle zur Senke. Eine Kante außerhalb jedes Weges wäre ein Nullspieler und würde jede Zählung über Betriebe aufblähen.
- Der Shapley-Wert gilt hier für die Gleichverteilung über alle Reihenfolgen der Betriebe; er wird über alle 2^k Koalitionen berechnet (Gleitkomma, nicht gesampelt), bis k = {C.K_MAX}.

**Grenzen**

- Ein Eigentümer je Kante, ein Gut, gleicher Erlös je Einheit, gerichtete Kanten ohne Kosten. Mehrfacheigentum an einer Kante, Kosten je Einheit, mehrere Güter und Netze mit Kreisen sind nicht gerechnet.
- Die Messreihen gelten für die genannten Netzformen (Zufalls-DAG, Reihen-Parallel-Netz, Engpasskette), die Kapazitätsbereiche und die Zahl der Netze je Zelle; ein anderer Netzerzeuger kann andere Anteile liefern.
- Die Hülle der minimalen Schnitte wird nur in einer Stichprobe von Richtungen mit dem Kern verglichen; „gleich“ heißt dort: in diesen Richtungen kein Unterschied.
- Aussagen über den Kern sind Aussagen über das Modell. Die Demo macht keine Rechts-, Anlage- oder Verhandlungsberatung; Betriebe, Kapazitäten und Erlöse sind eine Kulisse für das Rechenspiel.

**Bekannte Probleme und Unsicherheiten**

- Die Programme (HiGHS) liefern Gleitkommawerte; angezeigt wird gerundet, Kern-Grenzen gelten bis etwa 1e-7. Die Prüfung „Shapley im Kern“ nutzt die Toleranz 1e-9.
- Die Zeilenerzeugung bis k = {C.K_BIG_MAX} stützt sich auf ein gemischt-ganzzahliges Trennungsproblem; meldet der Löser die Zeitgrenze oder einen Fehler, steht das als Status da (Zeitgrenze und Löserfehler getrennt) und die Zahl gilt nicht als bewiesen.
- Die Tabellen der Experimente sind vorgerechnet (Datei `data/messreihen.json`, mit dem Hash der Quelltexte); die Kurzläufe in der App nutzen weniger Netze und dürfen von den Tabellen abweichen.
- Nur an erzeugten Netzen und sechs Handbeispielen geprüft, an keinem realen Verkehrsnetz. Ein echter Mausklick auf die Diagramme ist nicht getestet (die Diagramme sind nicht anklickbar).
- Die Literaturangaben sind nach Autoren, Jahr, Zeitschrift und Seiten über Bibliographien bestätigt; Volltexte wurden nicht gelesen.
"""

LITERATUR_HEAD = "**Literatur** (Autoren, Jahr, Zeitschrift und Seiten über Bibliographien bestätigt; Volltexte nicht gelesen)"
LITERATUR = [
    "- Ford, L. R., Fulkerson, D. R. (1956): Maximal Flow Through a Network. Canadian Journal of Mathematics 8, 399–404 (Max-Flow-Min-Cut).",
    "- Shapley, L. S. (1953): A Value for n-Person Games. In: Kuhn, Tucker (Hrsg.), Contributions to the Theory of Games II, Princeton University Press, 307–317.",
    "- Kalai, E., Zemel, E. (1982): Totally Balanced Games and Games of Flow. Mathematics of Operations Research 7(3), 476–478. Gelesen wurde nur die Zusammenfassung: Flussspiele fallen mit den total ausgewogenen Spielen zusammen.",
]

HOW_IT_WORKS = """\
1. **Netz und Eigentum:** Jede Kante gehört einem Betrieb (A, B, C ...), ihre Kapazität steht an der Kante. Der Wert einer Koalition S ist der größte Fluss von der Quelle s zur Senke t, den nur die Kanten aus S durchsetzen: v(S).
2. **Aufteilen:** Gesucht ist eine Zahlung x an die Betriebe mit x(N) = v(N). Der **Kern** besteht aus allen Zahlungen, bei denen keine Koalition mit ihrer eigenen Strecke mehr hätte (x(S) ≥ v(S)); der **Shapley-Wert** ist der durchschnittliche Grenzbeitrag über alle Reihenfolgen.
3. **Warum es einen Kern gibt:** Nach Max-Flow-Min-Cut ist v(N) die Kapazität eines kleinsten Schnitts. Zahlt man jeder Kante des Schnitts ihre Kapazität und allen anderen nichts, hat keine Koalition einen Anreiz abzuweichen: der Kern ist nie leer.
"""

TERMS_TABLE = """\
| Begriff | Bedeutung | Beispiel „Drei Kanten“ (A: 4 in Reihe mit B: 3 und C: 5) |
|---|---|---|
| Fluss | Einheiten, die je Tag von s nach t gelangen, ohne eine Kapazität zu überschreiten | A, B, C: 4 Einheiten, davon 3 über B und 1 über C |
| Schnitt | Kanten, deren Wegfall s von t trennt; seine Kapazität ist die Summe | {A}: Kapazität 4, {B, C}: Kapazität 8 |
| Koalition, v(S) | Betriebe, die nur ihre eigenen Kanten nutzen; Wert = größter Fluss darüber | v({A, C}) = 4, v({B, C}) = 0 (ohne A kommt nichts aus der Quelle) |
| Kern | Zahlungen mit x(N) = v(N) und x(S) ≥ v(S) für jede Koalition | A bekommt 3 bis 4, B nichts, C den Rest |
| Shapley-Wert | Mittel des Grenzbeitrags über alle Reihenfolgen | A: 5/2, B: 1/2, C: 1 |
| Veto-Kante | ohne sie ist v(N ohne sie) = 0 | A |
| überzählige Kante | kein Nullspieler, aber ohne sie sinkt v(N) nicht | B (C reicht allein neben A) |
"""

BEWEIS_H4 = r"""
**Warum eine überzählige Kante den Shapley-Wert aus dem Kern treibt.** Sei $i$ überzählig, also $v(N \setminus \{i\}) = v(N)$. Für jedes $x$ im Kern gilt $x(N \setminus \{i\}) \ge v(N \setminus \{i\}) = v(N) = x(N)$, also $x_i \le 0$, und wegen $x_i \ge v(\{i\}) \ge 0$ folgt $x_i = 0$. Der Shapley-Wert ist dagegen positiv, sobald $i$ in irgendeiner Reihenfolge einen positiven Beitrag leistet (das tut jede Kante auf einem Weg, die auf diesem Weg gebraucht wird). Beispiel „Drei Kanten“: B bekommt im Kern 0, im Shapley-Wert 1/2.
"""
