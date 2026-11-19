"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Rechenbudget (DEMO-PLAYBOOK 1c): die Koalitionswerte brauchen 2^k Einträge (Speicher O(2^k), Zeit O(Schnitte · 2^k)), der exakte Shapley-Wert
O(k · 2^k), das volle Kernprogramm 2^k - 2 Zeilen bei k Spalten (nur bis CORE_LP_MAX), die Zeilenerzeugung wenige Dutzend Zeilen. Die Zahlen am
Regler-Maximum (k = 20) stehen in tests/test_claims.py nicht als Zeit, sondern als Timeout-Test (tests/test_algorithm.py)."""

# ----------------------------------------------------------------------------------------------------------------------------- Regler
EXAMPLES = (
    "Reihe (5, 2, 3)",
    "Parallel (2, 3)",
    "Drei Kanten (4 | 3, 5)",
    "Brücke mit Einheitskapazität",
    "Brücke (3, 2, 1, 2, 3)",
    "Vier Kanten (5, 1 | 3, 5)",
)
RANDOM_FORMS = ("Zufalls-DAG", "Reihen-Parallel-Netz", "Engpasskette")
FORMS = EXAMPLES + RANDOM_FORMS
DEFAULT_FORM = "Drei Kanten (4 | 3, 5)"

CAPMODES = {"alle 1": (1, 1), "1 bis 5": (1, 5), "1 bis 20": (1, 20)}
DEFAULT_CAPMODE = "1 bis 5"

K_MIN, K_MAX, DEFAULT_K = 2, 20, 6      # Kantenzahl der Zufallsnetze; K_MAX = 20: 2^20 Koalitionen (Speicher 8 MB je Wertearray)
SEED_MAX, DEFAULT_SEED = 999999, 35

SP_MAX_INNER_NODES = 10                 # Reihen-Parallel-Netze: höchstens so viele innere Knoten (Schnittaufzählung 2^10)
K_BIG_MIN, K_BIG_MAX, DEFAULT_K_BIG = 10, 40, 30   # Größe-Tab: Zeilenerzeugung bis 40 Kanten (2^40 Koalitionen, nie aufgezählt)
BIG_NODES = 12                          # Knotenzahl der großen Zufalls-DAG im Größe-Tab (10 innere Knoten: 1 024 Schnitte)

# Obergrenzen der Anzeigen, die über alle Koalitionen oder mehrere Programme laufen (Spielraum: 2k Programme)
CONVEX_CHECK_MAX = 14                   # Konvexitätsprüfung O(k^2 · 2^k)
RANGES_LIVE_MAX = 12                    # Spielraum je Kante (2k Programme) ohne Knopf bis hierher
AUSBAU_MAX = 10                         # Ausbau-Tab: 2 Programme je Kapazitätsschritt
CAP_SWEEP = tuple(range(1, 9))          # Kapazitäten der Ausbau-Linie (wie in der Messreihe E3)
CAP_MAX = 20                            # größte Kapazität im Eingabefeld (0 = Kante gestrichen)

# ------------------------------------------------------------------------------------------------------------------------- Messreihen
SWEEP_SEEDS = (11, 12, 13, 14)          # feste Zufalls-Seeds der Messreihen: nie ein einzelner Seed als Befund
LIVE_RUNS = 12                          # Instanzen je Zelle im Kurzlauf der App

# ----------------------------------------------------------------------------------------------------------------------------- Farben
OWNER_COLORS = ("#4c78a8", "#f58518", "#54a24b", "#b279a2", "#9d755d", "#e45756", "#72b7b2", "#eeca3b", "#79706e", "#9ecae9")
COLOR_CUT = "#d62728"
COLOR_GRAY = "rgba(150,150,150,0.55)"
COLOR_LINE = "#4c78a8"
COLOR_LINE2 = "#f58518"

# ----------------------------------------------------------------------------------------------------------------------------- Presets
PRESETS = {
    "Drei Kanten (Standard)": {"form": "Drei Kanten (4 | 3, 5)", "k": DEFAULT_K, "capmode": DEFAULT_CAPMODE, "seed": DEFAULT_SEED},
    "Reihe (5, 2, 3)": {"form": "Reihe (5, 2, 3)", "k": DEFAULT_K, "capmode": DEFAULT_CAPMODE, "seed": DEFAULT_SEED},
    "Parallel (2, 3)": {"form": "Parallel (2, 3)", "k": DEFAULT_K, "capmode": DEFAULT_CAPMODE, "seed": DEFAULT_SEED},
    "Brücke mit Einheitskapazität": {"form": "Brücke mit Einheitskapazität", "k": DEFAULT_K, "capmode": DEFAULT_CAPMODE, "seed": DEFAULT_SEED},
    "Brücke (3, 2, 1, 2, 3)": {"form": "Brücke (3, 2, 1, 2, 3)", "k": DEFAULT_K, "capmode": DEFAULT_CAPMODE, "seed": DEFAULT_SEED},
    "Vier Kanten (5, 1 | 3, 5)": {"form": "Vier Kanten (5, 1 | 3, 5)", "k": DEFAULT_K, "capmode": DEFAULT_CAPMODE, "seed": DEFAULT_SEED},
    "Zufallsnetz mit 8 Kanten": {"form": "Zufalls-DAG", "k": 8, "capmode": "1 bis 5", "seed": 33},
    "Zufallsnetz ohne überzählige Kante": {"form": "Zufalls-DAG", "k": 6, "capmode": "1 bis 5", "seed": 38},
}
PRESET_HELP = {
    "Drei Kanten (Standard)": "Kante A (Kapazität 4) in Reihe mit den parallelen Kanten B (3) und C (5). Der kleinste Fall, in dem der Shapley-Wert den Kern verlässt.",
    "Reihe (5, 2, 3)": "Drei Kanten hintereinander: nur der Engpass (Kapazität 2) bestimmt den Wert. Jede Kante ist ein Veto-Spieler.",
    "Parallel (2, 3)": "Zwei Kanten nebeneinander: der Wert ist die Summe, jeder bekommt seine eigene Kapazität. Kein Konflikt, Shapley-Wert gleich Kern.",
    "Brücke mit Einheitskapazität": "Das Wheatstone-Netz mit fünf Kanten gleicher Kapazität; über die Brücke selbst fließt in einem größten Fluss nichts.",
    "Brücke (3, 2, 1, 2, 3)": "Dasselbe Netz mit ungleichen Kapazitäten: der Kern ist größer als die konvexe Hülle der minimalen Schnitte.",
    "Vier Kanten (5, 1 | 3, 5)": "Zwei Parallelgruppen hintereinander, jede Kante hat positiven Grenzbeitrag: der Shapley-Wert verletzt den Kern trotzdem (Restfall ohne überzählige Kante).",
    "Zufallsnetz mit 8 Kanten": "Zufalls-DAG mit 8 Kanten auf s-t-Wegen und Kapazitäten 1 bis 5.",
    "Zufallsnetz ohne überzählige Kante": "Zufalls-DAG mit 6 Kanten, in dem jede Kante gebraucht wird (kein Grenzbeitrag 0).",
}
