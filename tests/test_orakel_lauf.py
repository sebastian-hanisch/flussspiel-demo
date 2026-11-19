"""Der nachrechenbare Orakel-Lauf `tools/orakel_lauf.py`: die Zahlen der README (Abschnitt „Messung dieser README“) stehen hier als Behauptung.

Netzmenge: sechs Handbeispiele und 3 Netzformen × 3 Kapazitätsmodi × 7 Kantenzahlen (2 bis 8) × 6 Zufalls-Seeds (1 bis 6); Laufzeit rund 15 Sekunden."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("networkx")

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("orakel_lauf", ROOT / "tools" / "orakel_lauf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_oracle_run_reproduces_the_readme_numbers():
    res = _load().run()
    assert res == {"nets": 384, "coalitions": 27532, "cut_deviations": 0, "shapley_nets": 276, "shapley_deviations": 0}


def test_the_oracle_run_detects_a_wrong_value(monkeypatch):
    """Ohne Gegenprobe wäre „0 Abweichungen“ wertlos: ein falscher networkx-Wert muss als Abweichung gezählt werden."""
    mod = _load()
    nets = mod.net_set()[:6]
    monkeypatch.setattr(mod, "nx_value", lambda net, mask: -1.0)
    assert mod.run(nets)["cut_deviations"] == sum(1 << n.m for n in nets)
