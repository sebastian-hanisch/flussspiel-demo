"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Wirkung der Beispiele (Hilfetexte gegen die Rechnung)."""

import fsp_constants as C
import fsp_evaluation as ev
import fsp_game as G
import fsp_net as N
import fsp_presets as P
import fsp_scenario as S


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, preset in C.PRESETS.items():
        assert preset["form"] in C.FORMS and preset["capmode"] in C.CAPMODES, name
        assert C.K_MIN <= preset["k"] <= C.K_MAX and 0 <= preset["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Drei Kanten (Standard)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_choices_match_the_constants():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert P.SETTING_SPECS["form_select"].choices == C.FORMS and P.SETTING_SPECS["capmode_select"].choices == tuple(C.CAPMODES)


def test_example_presets_name_the_example_net_and_each_example_has_a_preset():
    forms = {p["form"] for p in C.PRESETS.values() if p["form"] in C.EXAMPLES}
    assert forms == set(C.EXAMPLES)


def test_effective_settings_ignore_k_capmode_and_seed_for_examples_only():
    a = ev.Settings(form="Reihe (5, 2, 3)", k=17, capmode="1 bis 20", seed=999)
    assert a.effective() == ev.Settings(form="Reihe (5, 2, 3)")
    b = ev.Settings(form="Zufalls-DAG", k=17, capmode="1 bis 20", seed=999)
    assert b.effective() == b


# -------------------------------------------------------------------------------------------- Wirkung der Beispiele gegen die Hilfetexte
def _a(name):
    return ev.analyse(ev.Settings(**C.PRESETS[name]).effective())


def test_help_texts_match_the_nets():
    a = _a("Reihe (5, 2, 3)")
    assert a.veto == (0, 1, 2) and a.v_grand == 2                          # „Jede Kante ist ein Veto-Spieler“
    assert _a("Parallel (2, 3)").in_core                                    # „Shapley-Wert gleich Kern“
    assert _a("Drei Kanten (Standard)").surplus == (1,) and not _a("Drei Kanten (Standard)").in_core
    b = _a("Brücke mit Einheitskapazität")
    assert b.surplus == (2,) and b.v_grand == 2                              # die Brücke trägt keinen zusätzlichen Fluss
    assert N.max_flow(b.net)[1][2] == 0
    v = _a("Vier Kanten (5, 1 | 3, 5)")
    assert v.surplus == () and not v.in_core                                 # Restfall: alle Grenzbeiträge positiv
    assert all(g > 0 for g in G.grand_marginals(v.v))


def test_bridge_preset_with_unequal_capacities_has_a_core_larger_than_the_hull():
    import fsp_core as K
    from fsp_rng import SplitMix64

    net = S.example_net("Brücke (3, 2, 1, 2, 3)")
    v = N.coalition_values(net)
    gap = K.hull_gap(net, v, K.directions(net.m, 40, SplitMix64(5)))
    assert not gap["equal"] and gap["with_gap"] > 0


def test_random_presets():
    eight = _a("Zufallsnetz mit 8 Kanten")
    assert eight.m == 8 and eight.surplus and eight.veto and not eight.in_core
    clean = _a("Zufallsnetz ohne überzählige Kante")
    assert clean.m == 6 and clean.surplus == ()                              # jede Kante wird gebraucht
