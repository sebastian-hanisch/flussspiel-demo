# Flussspiel – wem gehört die Strecke – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-flussspiel-demo.streamlit.app/)**

Demo der Linie [Koalitionen und Teilen](https://sebastianhanisch.net/konzepte-koalitionen-teilen.html) im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Vorgänger in der Linie sind das Zuordnungsspiel (`zuordnungsspiel-demo`, die Dualität eines
linearen Programms liefert dort den Kern) und der Machtindex (`machtindex-demo`, dort der Veto-Spieler als Machtfall); beide stehen live unter [machtindex-demo](https://sebastianhanisch-machtindex-demo.streamlit.app/) und [zuordnungsspiel-demo](https://sebastianhanisch-zuordnungsspiel-demo.streamlit.app/).
Verwandt sind [shapley-demo](https://sebastianhanisch-shapley-demo.streamlit.app/) und [kern-demo](https://sebastianhanisch-kern-demo.streamlit.app/) (dieselben Begriffe am Tourenspiel) und die Netzfluss-Demos
[network-flow-demo](https://sebastianhanisch-network-flow-demo.streamlit.app/), [dinic-demo](https://sebastianhanisch-dinic-demo.streamlit.app/) und
[gomory-hu-demo](https://sebastianhanisch-gomory-hu-demo.streamlit.app/): sie rechnen Flüsse und Schnitte ohne Spieler und ohne Aufteilung; hier gehört jede Kante einem Betrieb, und der Fluss wird aufgeteilt.

## Kernfrage

*Betriebe besitzen Strecken eines Netzes, der Wert einer Koalition ist der Fluss, den sie allein durchsetzen kann: wer bekommt was, und was sagt der Kern über Engpass, Redundanz und Ausbau?*
Die App beantwortet sie live für das gewählte Netz (Verdikt „Liegt der Shapley-Wert im Kern?“) und zeigt dazu Netzbild, Kern, Spielraum je Kante, Ausbau und die Zeilenerzeugung bis 40 Kanten.

## Modell

Gerichtetes Netz ohne Kreise, Quelle 0, Senke n−1; jede Kante gehört genau einem Betrieb und hat eine ganzzahlige Kapazität (Einheiten je Tag). Der Wert v(S) einer Koalition S ist der größte Fluss über
die Kanten aus S allein; jede Einheit bringt allen denselben Erlös (Modellannahme). Das Spiel ist ein **Gewinnspiel**: im Kern gilt x(S) ≥ v(S) für jede Koalition und x(N) = v(N); die Prüfung über alle
Koalitionen lässt die leere und die große Koalition aus. Formeln: Expander „📐 Mathematische Formulierung“ der App (Lesehilfe mit Symboltabelle, je Gleichung ein eigener Block).

Sechs feste Beispielnetze (von Hand gerechnet, die Werte stehen in `tests/test_claims.py`):

- Reihe (5, 2, 3): v(N) = 2, Shapley-Wert (2/3, 2/3, 2/3). Drei Kanten hintereinander, jede ist ein Veto-Spieler.
- Parallel (2, 3): v(N) = 5, Shapley-Wert (2, 3). Der Wert ist die Summe, jeder bekommt seine Kapazität.
- Drei Kanten (4 | 3, 5): v(N) = 4, Shapley-Wert (5/2, 1/2, 1). A mit Kapazität 4 liegt in Reihe mit den parallelen B (3) und C (5); Kern = {(x_A, 0, 4 − x_A) : 3 ≤ x_A ≤ 4}. B ist überzählig (x(A, C) = 7/2 < v(A, C) = 4).
- Brücke mit Einheitskapazität: v(N) = 2, Shapley-Wert (8/15, 9/20, 1/30, 9/20, 8/15). Das Wheatstone-Netz mit fünf Kanten; die Brücke bekommt im Kern 0.
- Brücke (3, 2, 1, 2, 3): v(N) = 5, Shapley-Wert (4/3, 1, 1/3, 1, 4/3), im Kern.
- Vier Kanten (5, 1 | 3, 5): v(N) = 6, Shapley-Wert (17/6, 1/2, 5/6, 11/6). Der Shapley-Wert verletzt den Kern, obwohl jede Kante einen positiven Grenzbeitrag hat: x(A, D) = 14/3 < v(A, D) = 5, Verletzung 1/3.

Dazu drei Zufallsnetz-Formen (Zufalls-DAG, Reihen-Parallel-Netz, Engpasskette), nur mit Kanten an s-t-Wegen (andernfalls Nullspieler, siehe „Befunde und Korrekturen“), Zufall über SplitMix64 (`fsp_rng.py`, wörtliche Kopie).

## Methodik

- **Zwei Rechenwege für v(S):** kürzeste Erweiterungspfade (Edmonds-Karp, für eine Koalition und für die Flussanzeige) und Max-Flow-Min-Cut: v(S) = min über alle s-t-Schnitte C von c(C ∩ S), vektorisiert über alle 2^k Koalitionen (dominierte Schnitte entfallen).
- **Regler:** Kantenzahl k von 2 bis 20 bei Zufallsnetzen (alle 2^k Koalitionswerte), im Reiter Größe 10 bis 40 Kanten.
- **Shapley-Wert** über alle 2^k Koalitionen (Gewichtsformel, Gleitkomma), bis k = 20; bis 12 Kanten zusätzlich als Brüche. Konvexitätsprüfung bis 14 Kanten.
- **Kern:** Linearprogramm über alle Koalitionen (HiGHS, bis 14 Kanten); darüber und für große Netze **Zeilenerzeugung** mit Trennung durch Aufzählung (bis 20 Kanten) oder durch ein gemischt-ganzzahliges Fixkostenflussproblem (`scipy.optimize.milp`, bis 40 Kanten). Der Löserstatus wird ausgewiesen; ein „optimal“ steht nur, wenn die Trennung ohne Zeitgrenze und bis zur Toleranz 1e-7 abgeschlossen ist. Der Reiter Ausbau rechnet bis 10 Kanten (zwei Programme je Kapazitätsschritt), die Hülle-Prüfung läuft bis 10 Kanten ohne Knopf.
- **Überzählig, Veto, Nullspieler, Konvexität** aus den Koalitionswerten (Definitionen in den Orakeltests in Brute Force nachgebildet).
- **Messreihen** (Experimente 1 bis 5) sind vorgerechnet (`tools/make_messreihen.py` → `data/messreihen.json`, mit dem Hash der Quelltexte); die App rechnet nur Kurzläufe live.

## Befunde (gemessen, keine Behauptungen)

Alle Messreihen: Netze aus SplitMix64 mit den festen Zufalls-Seeds 11, 12, 13, 14; Anteil als Mittel über die vier Zufalls-Seeds; „im Kern“ heißt Überschuss ≥ −1e−9. Alle Zahlen stehen in `tests/test_claims.py`
(aus `data/messreihen.json`; `tests/test_measurements.py` prüft den Hash und rechnet kleine Zellen neu).

| Frage | Befund (Voraussetzung, Zahl der Läufe) | Test |
|---|---|---|
| Wie oft liegt der Shapley-Wert im Kern, wenn das Netz wächst? (Experiment 1) | 4 472 Netze in 50 Zellen (je 100 Netze bis k = 14, je 12 für k = 16, 18, 20). Zufalls-DAG, Kapazität 1: k = 3: 59 %, k = 4: 31 %, k = 5: 22 %, k = 6: 11 %, k = 8: 4 %, k = 10: 0 %. Zufalls-DAG, Kapazitäten 1 bis 5: k = 3: 67 %, k = 4: 45 %, k = 5: 30 %, k = 6: 15 %, k = 8: 0 %. Im Zufalls-DAG gab es in dieser Messreihe ab k = 12 keinen Treffer (k = 12, 14: je 100 Netze; k = 16 bis 20: je 12 Netze; beide Kapazitätsmodi). | `test_e1_cells_and_sentences`, `test_e1_in_the_core_values_by_the_numbers_of_the_readme` |
| Gilt das in allen Netzformen? | In allen sechs Reihen (3 Netzformen × 2 Kapazitätsmodi): bei k = 3 mindestens 59 %, bei k = 12 höchstens 9 %. Reihen-Parallel-Netz, Kapazität 1: k = 3: 72 %, k = 12: 2 %. Engpasskette, Kapazität 1: k = 3: 75 %, k = 10: 2 %, k = 12: 9 %. In dieser Messreihe nicht monoton, aber kein gesicherter Befund (der Unterschied beträgt etwa zwei Standardfehler): Engpasskette mit Kapazität 1: 2 % bei k = 10, aber 9 % bei k = 12. | `test_e1_over_all_six_series_start_high_and_end_low`, `test_e1_the_series_of_bottleneck_chains_is_not_monotone` |
| Gibt es einen Fall mit überzähliger Kante und Shapley-Wert im Kern? | Nein: in keiner der 50 Zellen (4 472 Netze) zugleich (Beweis in der App, Zählung hier). | `test_e1_surplus_and_in_the_core_never_together`, `tests/test_properties.py` |
| Wie häufig ist eine überzählige Kante? | Zufalls-DAG, Kapazität 1: 41 % bei k = 3 und 100 % ab k = 12 (100 Netze je Zelle, 12 ab k = 16). | `test_e1_share_of_nets_with_a_surplus_edge` |
| Ist der Kern die Hülle der minimalen Schnitte? (Experiment 2) | Anteil der Netze mit Kern gleich Hülle in allen getesteten Richtungen: Kapazität 1: 100 %, Kapazitäten 1 bis 5: 24 %, Kapazitäten 1 bis 20: 19 %; je 100 Netze (Zufalls-DAG, k = 4 bis 8), 40 Zufallsrichtungen und die 2k Koordinatenrichtungen je Netz (48 bis 56 Richtungen). Eine Stichprobe von Richtungen, kein Beweis für alle. | `test_e2_hull_equals_core_shares` |
| Welchen Spielraum hat eine Kante im Kern? | Zufalls-DAG, k = 4 bis 8, je 80 Netze je Kapazitätsmodus, Kapazität 1 und 1 bis 5: Veto-Kanten (17 und 15 Kanten) Kern-Obergrenze gleich v(N); überzählige Kanten (234 und 171 Kanten) Obergrenze 0 in allen; übrige Kanten Kern-Obergrenze im Mittel 0,40 und 0,35 (229 und 294 Kanten). 5 von 294 übrigen Kanten bei Kapazitäten 1 bis 5 haben die Obergrenze 0. | `test_e2_range_classes` |
| Steigt die Zahlung, wenn die eigene Kapazität steigt? (Experiment 3) | Kapazität einer Kante von 1 bis 8, Zufalls-DAG, Kapazitäten 1 bis 5: in 304 Kante-Netz-Paaren (k = 5, 6, 8): Shapley-Wert fällt in 0, die Kern-Untergrenze in 50, die Kern-Obergrenze in 24 Paaren. Beispiel: Kern-Untergrenze 1 für die Kapazitäten 1 bis 4, 0 ab 5, Shapley-Wert 2/3 bis 5/2. | `test_e3_capacity_monotonicity`, `test_e3_the_example_of_the_plan_is_recomputed` |
| Wann liegt der Shapley-Wert außerhalb, obwohl keine Kante überzählig ist? (Experiment 5) | Zufalls-DAG, k = 3 bis 8, Kapazitäten 1 bis 5: 600 Netze: 141 konvexe Spiele, alle 141 mit Shapley-Wert im Kern; 185 Netze ohne überzählige Kante, in 22 davon liegt er außerhalb, alle 22 nicht konvex. | `test_restfall_counts` |
| Ist der Kern auch in allen Teilspielen nicht leer? | 0 leere Kerne in 3 776 Teilspielen (64 Netze, k = 4 bis 7, Zufalls-DAG, Kapazitäten 1 bis 5; Teilspiele mit einem Spieler sind trivial). | `test_total_balancedness_over_all_subgames` |
| Wie viele Zeilen braucht der Kern bei großen Netzen? (Experiment 4) | Größte Zahlung einer zufällig gewählten Kante eines minimalen Schnitts, Zufalls-DAG, Kapazitäten 1 bis 5, alle Läufe bewiesen optimal: m = 40: 58 bis 145 Zeilen (6 Netze, 2^40 Koalitionen), m = 30: 42 bis 79 Zeilen (6 Netze), m = 20: 25 bis 26 Zeilen (3 Netze); gleich dem vollen Programm in 14 von 14 Netzen (m = 10, 12, 14). Aufzählungs- und gemischt-ganzzahlige Trennung: gleicher Zielwert in 3 von 3 Netzen (m = 14, 14, 20). | `test_e4_rows_of_the_row_generation`, `test_e4_milp_separation_equals_the_enumeration_on_larger_nets` |
| Wie oft brauchte der Netzerzeuger die Auffüllung? | In 95 geprüften Netzen (alle k von 2 bis 20, 5 Zufalls-Seeds) brauchte der Erzeuger die Auffüllung nie. | `test_the_dag_generator_never_needed_the_fill_in_on_the_checked_nets` |

**Messung dieser README, nicht im Test:** Auf dem Entwicklungsrechner dauerte die Berechnung aller 2^k Koalitionswerte mit Shapley-Wert bei k = 20 etwa 0,3 bis 1 Sekunde je nach Rechnerlast (ein Netz, ein Rechner,
keine Zeit-Assertion; der Worst-Case-Test `test_worst_case_at_the_slider_maximum_completes` prüft nur, dass der Lauf endet). Die Zeilenerzeugung bei m = 40 brauchte je nach Rechnerlast eine bis rund neun Sekunden je Netz.
Nachrechenbar im Repo: `python tools/orakel_lauf.py` (rund 15 Sekunden) prüft an einer festen Netzmenge (sechs Handbeispiele und 3 Netzformen × 3 Kapazitätsmodi × Kantenzahlen 2 bis 8 × Zufalls-Seeds 1 bis 6, zusammen 384 Netze) die Schnittformel gegen networkx, scipy und den eigenen Max-Flow auf jeder Koalition (27 532 Koalitionen, 0 Abweichungen) und den Shapley-Wert gegen die Aufzählung aller Reihenfolgen in Brüchen (276 Netze mit höchstens 6 Kanten, bis zu 720 Reihenfolgen je Netz, 0 Abweichungen); `tests/test_orakel_lauf.py` hält diese Zahlen fest. Die übrigen Orakel (minimale Schnitte gegen Aufzählung, Kern gegen Zeilenerzeugung) laufen in `tests/test_oracle_flussspiel.py` auf kleineren Netzmengen. Die früheren Scratch-Zahlen (567 Netze, 128 772 Koalitionen) sind nicht übernommen, weil sich ihre Netzmenge nicht rekonstruieren lässt.

### Hypothesen und Stand

| Nr. | Hypothese | Stand |
|---|---|---|
| H1 | Der Kern ist nie leer, auch in allen Teilspielen | Ja: Schnitt-Auszahlung (Max-Flow-Min-Cut) und 0 leere Kerne in 3 776 Teilspielen |
| H2 | Schnitt-Auszahlungen minimaler Schnitte liegen im Kern | Ja: Orakel und Eigenschaftstest (`tests/test_properties.py`) |
| H3 | Bei Einheitskapazität ist der Kern die Hülle der minimalen Schnitte, sonst größer | In den getesteten Richtungen bei Einheitskapazität Kern gleich Hülle (100 %), bei ungleichen Kapazitäten meist größer (Gleichheit in 24 % bzw. 19 % der Netze; Experiment 2). Für Einheitskapazität ist das ein bekannter Satz (nicht im Volltext geprüft) |
| H4 | Überzählige Kante ⇒ Shapley-Wert außerhalb des Kerns | Ja: Beweis in der App, 0 Gegenfälle in 4 472 Netzen und im Eigenschaftstest |
| H5 | Der Anteil „Shapley im Kern“ fällt mit k auf 0 | Der Anteil fällt mit k stark: im Zufalls-DAG 0 % ab k = 10 bzw. 8, in den beiden anderen Netzformen bis k = 12 auf höchstens 9 % (nicht gesichert monoton) |
| H6 | Flussspiele haben den Shapley-Wert meist im Kern (Plan: 75 %) | Nein für die Zufallsnetze dieser Demo ab k = 4 bis 5; die 75 % der Vormessung sind vermutlich durch Nullspieler (kleine wirksame Kantenzahl) erklärt, nicht nachgerechnet, siehe unten |
| H7 | Flussspiel und Zuordnungsspiel unterscheiden sich in der Häufigkeit | In dieser Demo nicht gemessen (Vormessung als Wegwerfskript, nicht übernommen) |
| H8 | Ohne Überzähligkeit liegt der Shapley-Wert nur bei nichtkonvexem Spiel außerhalb; konvex ⇒ im Kern | Ja: 141 von 141 konvex im Kern, 22 von 22 nichtkonvex |
| H9 | Der Shapley-Wert ist monoton in der eigenen Kapazität, die Kern-Untergrenze nicht | Ja: 0 gegen 50 von 304; die Kern-Obergrenze fällt in 24 |
| H10 | Eine Veto-Kante kann den ganzen Wert beanspruchen | Ja: Beweis, Eigenschaftstest, Kern-Obergrenze gleich v(N) bei allen 17 und 15 Veto-Kanten |
| H11 | Der Kern lässt sich bis m = 40 per Zeilenerzeugung rechnen | Ja, für die gemessene Zielgröße und einen Netztyp (Zufalls-DAG) |

## Befunde und Korrekturen gegenüber dem Plan

1. **Die Zählung „Shapley im Kern 75 %“ der ersten Vormessung ist kein Befund über Flussspiele.** Dort lagen von neun Kanten nur ein bis sechs auf einem s-t-Weg; die übrigen waren Nullspieler. Hier zählen nur Kanten an Wegen,
   und der Anteil liegt bei k = 3 zwischen 59 und 84 % und bei k = 12 bei höchstens 9 %. Die 75 % selbst wurden im Bau nicht nachgerechnet (der damalige Erzeuger liegt nicht in der Demo).
2. **„Kern = Min-Cut-Dualität: die Eckpunkte“** gilt als Gleichheit mit der Hülle der minimalen Schnitte nur bei Einheitskapazität (100 %); bei Kapazitäten 1 bis 5 in 24 %, bei 1 bis 20 in 19 % der Netze.
3. **Kern-Obergrenze bei wachsender Kapazität:** der Plan erwartete, dass sie kaum fällt (1 von 176 Paaren in der Vormessung); mit k = 8 dazu und neuen Netzen fällt sie in 24 von 304 Paaren. Der Plan-Hook „Shapley steigt, die Kern-Untergrenze fällt“ bestätigt sich (0 gegen 50), die Obergrenze gehört dazu.
4. **Übrige Kanten mit Obergrenze 0:** 5 von 294 übrigen Kanten (Kapazitäten 1 bis 5) bekommen im Kern höchstens 0, obwohl sie weder überzählig noch Nullspieler sind (Grenzbeitrag zur großen Koalition positiv, aber jede Kern-Zahlung an sie ist 0). Das stand nicht im Plan.
5. **Anteile weichen leicht von der Vormessung ab** (zum Beispiel 59 statt 65 % bei k = 3 im Zufalls-DAG mit Kapazität 1): andere Netze (SplitMix64 statt numpy-Zufall), vier Zufalls-Seeds zu je 25 Netzen. Maßgeblich sind die Zahlen oben.
6. **Lücke zwischen Kern und Hülle:** der Plan nannte einen Median der Lücke pro Richtung; die Demo misst je Netz die größte Lücke relativ zu v(N). Beide Größen sind verschieden und nicht vergleichbar; die App nennt nur die Anteile (24 % und 19 %).
7. **Zahl der Knoten ist kein Regler:** sie folgt aus der Kantenzahl wie in der Messreihe (Knotenvorrat des Erzeugers: 6 bis k = 10, 7 bis k = 16, danach 8, nach dem Streichen unbenutzter Knoten oft weniger; Reihen-Parallel-Netze höchstens 10 innere Knoten). Bei einem festen Beispielnetz sind Kantenzahl, Kapazitätsmodus und Zufalls-Seed gesperrt.
8. **Kein Knopf für k = 17 bis 20:** die Rechnung ist schnell genug (etwa 0,3 bis 1 Sekunde bei k = 20); die App zeigt dort nur einen Spinner. Der Spielraum je Kante (2k Programme) läuft bis 12 Kanten ohne Knopf, darüber auf Knopfdruck.
9. **Hülle gegen Kern, Richtungen:** gleichverteilt auf [−1, 1] statt normalverteilt (SplitMix64 hat hier kein `normal()`).

## Ehrliche Grenzen

- Nur Netze ohne Kreise, ein Eigentümer je Kante, ein Gut, gleicher Erlös je Einheit, keine Kosten: siehe „Wo die Annahmen enden“ in der App.
- Die Messreihen gelten für den hier gebauten Netzerzeuger; ein anderer Erzeuger kann andere Anteile liefern (die Richtung „fällt mit k“ hielt in allen drei Netzformen).
- Die Hülle wird nur in Richtungen verglichen; die Zeilenerzeugung bei m > 20 hängt am gemischt-ganzzahligen Löser (HiGHS); bei Zeitgrenze oder Löserfehler steht „nicht bewiesen“.
- Nur an erzeugten Netzen und den sechs Beispielen geprüft, an keinem realen Verkehrsnetz. Der Vergleich mit dem Zuordnungsspiel steht in der Vormessung, nicht in dieser Demo.

## Annahmen, Grenzen und bekannte Probleme

**Annahmen**

- Ein gerichtetes Netz ohne Kreise mit Quelle und Senke. Jede Kante gehört genau einem Betrieb und hat eine ganzzahlige Kapazität (Einheiten je Tag). Ein Betrieb ist ein Spieler, ein Spieler besitzt genau eine Kante.
- Jede durchgesetzte Einheit bringt allen denselben Erlös; der Wert einer Koalition ist der größte Fluss, den ihre Kanten allein durchsetzen. Das ist eine Modellannahme, kein Befund über reale Netze.
- Der Wert ist ein Gewinn (Erlös), kein Kostenspiel: im Kern gilt x(S) ≥ v(S), keine Gruppe bekommt weniger als sie allein durchsetzt.
- Zufallsnetze enthalten nur Kanten an Wegen von der Quelle zur Senke. Eine Kante außerhalb jedes Weges wäre ein Nullspieler und würde jede Zählung über Betriebe aufblähen.
- Der Shapley-Wert gilt hier für die Gleichverteilung über alle Reihenfolgen der Betriebe; er wird über alle 2^k Koalitionen berechnet (Gleitkomma, nicht gesampelt), bis k = 20.

**Grenzen**

- Ein Eigentümer je Kante, ein Gut, gleicher Erlös je Einheit, gerichtete Kanten ohne Kosten. Mehrfacheigentum an einer Kante, Kosten je Einheit, mehrere Güter und Netze mit Kreisen sind nicht gerechnet.
- Die Messreihen gelten für die genannten Netzformen (Zufalls-DAG, Reihen-Parallel-Netz, Engpasskette), die Kapazitätsbereiche und die Zahl der Netze je Zelle; ein anderer Netzerzeuger kann andere Anteile liefern.
- Die Hülle der minimalen Schnitte wird nur in einer Stichprobe von Richtungen mit dem Kern verglichen; „gleich“ heißt dort: in diesen Richtungen kein Unterschied.
- Aussagen über den Kern sind Aussagen über das Modell. Die Demo macht keine Rechts-, Anlage- oder Verhandlungsberatung; Betriebe, Kapazitäten und Erlöse sind eine Kulisse für das Rechenspiel.

**Bekannte Probleme und Unsicherheiten**

- Die Programme (HiGHS) liefern Gleitkommawerte; angezeigt wird gerundet, Kern-Grenzen gelten bis etwa 1e-7. Die Prüfung „Shapley im Kern“ nutzt die Toleranz 1e-9.
- Die Zeilenerzeugung bis k = 40 stützt sich auf ein gemischt-ganzzahliges Trennungsproblem; meldet der Löser die Zeitgrenze oder einen Fehler, steht das als Status da (Zeitgrenze und Löserfehler getrennt) und die Zahl gilt nicht als bewiesen.
- Die Tabellen der Experimente sind vorgerechnet (Datei `data/messreihen.json`, mit dem Hash der Quelltexte); die Kurzläufe in der App nutzen weniger Netze und dürfen von den Tabellen abweichen.
- Nur an erzeugten Netzen und sechs Handbeispielen geprüft, an keinem realen Verkehrsnetz. Ein echter Mausklick auf die Diagramme ist nicht getestet (die Diagramme sind nicht anklickbar).
- Die Literaturangaben sind nach Autoren, Jahr, Zeitschrift und Seiten über Bibliographien bestätigt; Volltexte wurden nicht gelesen.

## Tests

302 Tests (`pytest --collect-only -q tests`), lokal einmal gemessen rund 26 Sekunden mit `-n 4` (Windows, Rechner ohne weitere Last; mit parallelen Prozessen dauert es länger). Zusammensetzung:

- **Einheiten** (`test_algorithm.py`): Fluss, Schnitte, Koalitionswerte, Shapley-Wert, Kern, Zeilenerzeugung an den Mini-Instanzen von Hand (`conftest.py`), dazu ein Worst-Case-Lauf bei k = 20 in allen Netzformen.
- **Orakel** (`test_oracle_flussspiel.py`): Schnittformel gegen networkx und scipy auf jeder Koalition, Shapley-Wert gegen alle Reihenfolgen in Brüchen, Konvexität gegen die Definition, minimale Schnitte gegen Aufzählung, Kern gegen Prüfung aller Koalitionen, Zeilenerzeugung (Aufzählung und gemischt-ganzzahlig) gegen das volle Programm, totale Ausgewogenheit.
- **Eigenschaften** (`test_properties.py`, hypothesis): Effizienz, Nullspieler, überzählige Kante, Veto-Kante, Konvexität, Kapazitätsmonotonie des Shapley-Wertes, Schnittformel gegen Fluss.
- **Randfälle, Verdikte, Presets, Texte, Cache-Schlüssel** (`test_edge_cases.py`, `test_verdicts.py`, `test_presets.py`, `test_texts.py`, `test_cache_keys.py`): ein und zwei Kanten, Gleichstände, Kapazität 0, jedes Verdikt je Schwelle und jede Zahl im Text, jedes Preset.
- **Messreihen und Zahlen der README** (`test_measurements.py`, `test_claims.py`): Hash der Quelltexte, kleine Zellen neu gerechnet, jede README-Zahl.
- **Oberfläche** (`test_app.py`, `test_layout.py`, `test_footer.py`, `test_visualization.py`, `test_runtime_imports.py`): jeder Knopf (auch die ▶️-Knöpfe), jeder Regler an Minimum und Maximum, jede Netzform, jedes Preset, Permalink, Schnellstart und Hervorhebung, Footer, Abbildungen, Laufzeit-Importe.

Keine Zeit-Assertion; Zahlen aus Programmen (HiGHS) nur als Band oder mit Toleranz. Keine Mutationstests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Ergebnis, Kernfrage, fünf Reiter, fünf Experimente, Grenzen, Formeln |
| `fsp_net.py` | Netz, Max-Flow, Schnitte, Koalitionswerte (Schnittformel) |
| `fsp_game.py` | Shapley-Wert, Nullspieler, überzählige und Veto-Kanten, Konvexität, Kernprüfung |
| `fsp_core.py` | Kern: Programm, Spielraum, Schnitt-Auszahlungen, Hülle, Zeilenerzeugung (Aufzählung und gemischt-ganzzahlig) |
| `fsp_scenario.py` | Handbeispiele und Zufallsnetze (SplitMix64) |
| `fsp_rng.py` | SplitMix64, wörtliche Kopie von `ab-test-demo/abt_rng.py` |
| `fsp_experiments.py`, `fsp_tables.py` | Messroutinen und vorgerechnete Tabellen (`data/messreihen.json`, `tools/make_messreihen.py`) |
| `fsp_evaluation.py` | Kennzahlen und Verdikte (reine Funktionen) |
| `fsp_texts.py` | Texte, die App und README gemeinsam haben (Annahmen, Literatur, Begriffstabelle) |
| `fsp_visualization.py`, `fsp_pdf_export.py` | Plotly-Abbildungen, PDF (erst nach Klick) |
| `fsp_constants.py`, `fsp_presets.py`, `fsp_layout.py` | Regler-Grenzen, Presets und Permalink, fester Kopf |
| `tests/` | Einheiten, Orakel, Eigenschaften, Verdikte, Messreihen, Claims, App, Layout, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- Mehrfacheigentum an Kanten, Kosten je Einheit, mehrere Güter, Netze mit Kreisen.
- Sampling des Shapley-Wertes ab k = 21 (dafür [shapley-demo](https://sebastianhanisch-shapley-demo.streamlit.app/)); über alle Koalitionen wird bis k = 20 gerechnet.
- Der Nukleolus und andere Lösungskonzepte ([kern-demo](https://sebastianhanisch-kern-demo.streamlit.app/) zeigt ihn am Tourenspiel).
- Ein Nachbau des Zuordnungsspiels als Vergleich: das ist eine eigene Demo der Linie.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Cloud-Version (Python) richtet sich nach der Einstellung im Streamlit-Dashboard; die CI läuft mit Python `3.12`, die lokale Entwicklung mit `3.14`.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Koalitionen und Teilen: Macht und Aufteilung](https://sebastianhanisch.net/konzepte-koalitionen-teilen.html).
