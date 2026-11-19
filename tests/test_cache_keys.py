"""Cache-Schlüssel der App: die Analyse hängt nur von den wirksamen Reglern ab. Bei einem festen Beispielnetz zählen Kantenzahl, Kapazitätsmodus und Zufalls-Seed
nicht (sie werden vor dem Cache auf Standardwerte gesetzt): gleiche Rechnung mit dem Schlüssel wie mit allen Reglern, auch wenn die übrigen verstellt sind."""

import numpy as np
import pytest

import fsp_constants as C
import fsp_evaluation as ev

ALL = [ev.Settings(form=f, k=k, capmode=c, seed=s) for f in C.FORMS for (k, c, s) in ((C.DEFAULT_K, C.DEFAULT_CAPMODE, C.DEFAULT_SEED), (3, "alle 1", 5), (12, "1 bis 20", 77))]


@pytest.mark.parametrize("s", ALL, ids=[f"{s.form}-{s.k}-{s.capmode}-{s.seed}" for s in ALL])
def test_result_with_the_key_equals_result_with_all_settings(s):
    assert ev.net_of(s.effective()) == ev.net_of(s)
    a, b = ev.analyse(s.effective()), ev.analyse(s)
    assert a.net == b.net and np.array_equal(a.v, b.v) and a.in_core == b.in_core


def test_keys_ignore_exactly_the_irrelevant_regler():
    for form in C.EXAMPLES:
        assert ev.Settings(form=form, k=3, capmode="alle 1", seed=1).effective() == ev.Settings(form=form, k=20, capmode="1 bis 20", seed=999).effective()
    for form in C.RANDOM_FORMS:
        base = ev.Settings(form=form)
        for other in (ev.Settings(form=form, k=C.DEFAULT_K + 1), ev.Settings(form=form, capmode="alle 1"), ev.Settings(form=form, seed=C.DEFAULT_SEED + 1)):
            assert base.effective() != other.effective()
