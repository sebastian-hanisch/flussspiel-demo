"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

Alle Werte sind von Hand gerechnet (Max-Flow-Min-Cut, Shapley-Wert über alle Reihenfolgen), nicht aus dem Code abgeschrieben."""

import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fsp_net import Net  # noqa: E402


@pytest.fixture
def reihe():
    """Reihe (5, 2, 3): drei Kanten hintereinander. Der Fluss ist der Engpass 2; nur die Koalition aller drei Kanten setzt etwas durch."""
    return Net(4, ((0, 1, 5), (1, 2, 2), (2, 3, 3)), "Reihe")


@pytest.fixture
def parallel():
    """Parallel (2, 3): zwei Kanten s -> t. v additiv: v(A) = 2, v(B) = 3, v(AB) = 5."""
    return Net(2, ((0, 1, 2), (0, 1, 3)), "Parallel")


@pytest.fixture
def drei():
    """Drei Kanten: A = s -> 1 (4) in Reihe mit B = 1 -> t (3) und C = 1 -> t (5).
    v(A) = v(B) = v(C) = 0, v(AB) = 3, v(AC) = 4, v(BC) = 0, v(ABC) = 4. Kern = {(x_A, 0, 4 - x_A) : x_A >= 3}. Shapley = (5/2, 1/2, 1)."""
    return Net(3, ((0, 1, 4), (1, 2, 3), (1, 2, 5)), "Drei Kanten")


@pytest.fixture
def bruecke():
    """Wheatstone-Brücke mit Einheitskapazität: s -> 1, s -> 2, 1 -> 2 (Brücke), 1 -> t, 2 -> t. v(N) = 2; Shapley = (8/15, 9/20, 1/30, 9/20, 8/15)."""
    return Net(4, ((0, 1, 1), (0, 2, 1), (1, 2, 1), (1, 3, 1), (2, 3, 1)), "Brücke")


@pytest.fixture
def vier():
    """Vier Kanten (5, 1 | 3, 5): Gruppe 1 = A, B (s -> 1), Gruppe 2 = C, D (1 -> t). v(S) = min(Kapazität Gruppe 1 in S, Kapazität Gruppe 2 in S).
    v(N) = 6, Shapley = (17/6, 1/2, 5/6, 11/6); x(A, D) = 14/3 < v(A, D) = 5."""
    return Net(3, ((0, 1, 5), (0, 1, 1), (1, 2, 3), (1, 2, 5)), "Vier Kanten")


HAND_SHAPLEY = {
    "reihe": [F(2, 3)] * 3,
    "parallel": [F(2), F(3)],
    "drei": [F(5, 2), F(1, 2), F(1)],
    "bruecke": [F(8, 15), F(9, 20), F(1, 30), F(9, 20), F(8, 15)],
    "vier": [F(17, 6), F(1, 2), F(5, 6), F(11, 6)],
}
HAND_VALUE_GRAND = {"reihe": 2, "parallel": 5, "drei": 4, "bruecke": 2, "vier": 6}
