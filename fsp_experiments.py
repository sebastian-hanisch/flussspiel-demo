"""Messreihen der Demo: dieselben Routinen für die vorgerechneten Tabellen (`tools/make_messreihen.py` -> `data/messreihen.json`) und die Kurzläufe der App.

Jede Zelle (Netzform, Kapazitätsmodus, Kantenzahl k) bekommt mehrere Zufalls-Seeds (`C.SWEEP_SEEDS`) mit je einigen Netzen; angegeben wird das Mittel der
Anteile über die Seeds und ihre Standardabweichung, dazu die Zählungen. Die Netze eines Seeds stammen aus SplitMix64 (`instance_seed`)."""

import time

import numpy as np

import fsp_constants as C
import fsp_core as K
import fsp_game as G
import fsp_net as N
import fsp_scenario as S
from fsp_rng import SplitMix64

TOL = 1e-9                  # Kern-Toleranz der Zählungen (x(S) >= v(S) - TOL)
_MASK63 = (1 << 63) - 1


def instance_seed(salt: int, seed: int, k: int, j: int) -> int:
    """Zufalls-Seed des j-ten Netzes einer Zelle; verschiedene Salze trennen die Messreihen."""
    return (((salt * 1_000_003 + seed) * 1_009 + k) * 10_007 + j) & _MASK63


def _mean_std(values) -> tuple:
    a = np.asarray(values, dtype=float)
    return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), float("nan"))


def classify(net: N.Net, v=None) -> dict:
    """Kennzeichen eines Netzes für die Zählungen: Shapley im Kern, überzählige Kanten, Veto-Kanten, Konvexität (nur bis CONVEX_CHECK_MAX Kanten)."""
    v = N.coalition_values(net) if v is None else v
    phi = G.shapley(v)
    margin, gap, _ = G.core_margin(v, phi)
    surplus = G.surplus_edges(v)
    return {
        "in_core": bool(margin >= -TOL and gap <= 1e-7),
        "margin": margin,
        "surplus": len(surplus),
        "surplus_share_of_shapley": float(phi[surplus].sum() / v[-1]) if surplus and v[-1] > 0 else 0.0,
        "veto": len(G.veto_edges(v)),
        "convex": G.is_convex(v) if net.m <= C.CONVEX_CHECK_MAX else None,
        "v_grand": float(v[-1]),
    }


# ------------------------------------------------------------------------------------------------------------ E1: Shapley im Kern
def shapley_core_cell(form: str, capmode: str, k: int, runs_per_seed: int, seeds=C.SWEEP_SEEDS, salt: int = 1) -> dict:
    """Anteil der Netze mit Shapley-Wert im Kern (Mittel und Standardabweichung über die Seeds) und Zählungen zur überzähligen Kante."""
    rates, n_total, n_surplus, n_surplus_and_core, n_clean, n_clean_in_core = [], 0, 0, 0, 0, 0
    for sd in seeds:
        hits = 0
        for j in range(runs_per_seed):
            net = S.make_net(form, k, capmode, instance_seed(salt, sd, k, j))
            c = classify(net)
            hits += c["in_core"]
            n_total += 1
            if c["surplus"]:
                n_surplus += 1
                n_surplus_and_core += c["in_core"]
            else:
                n_clean += 1
                n_clean_in_core += c["in_core"]
        rates.append(hits / runs_per_seed)
    mean, std = _mean_std(rates)
    return {"form": form, "capmode": capmode, "k": k, "runs": n_total, "in_core_mean": mean, "in_core_std": std,
            "surplus_share": n_surplus / n_total, "surplus_and_in_core": n_surplus_and_core,
            "clean": n_clean, "clean_in_core": n_clean_in_core}


# --------------------------------------------------------------------------------------------------------- E2: Kern gegen Hülle
def hull_cell(capmode: str, ks, runs_per_k: int, seeds=C.SWEEP_SEEDS, n_dirs: int = 40, salt: int = 2) -> dict:
    """Anteil der Netze, in denen Kern und Hülle der minimalen Schnitte in allen getesteten Richtungen übereinstimmen (Mittel über Seeds)."""
    per_seed, gaps, n_total, n_cuts = [], [], 0, []
    for sd in seeds:
        equal, tot = 0, 0
        for k in ks:
            for j in range(runs_per_k):
                net = S.make_net("Zufalls-DAG", k, capmode, instance_seed(salt, sd, k, j))
                v = N.coalition_values(net)
                dirs = K.directions(net.m, n_dirs, SplitMix64(instance_seed(salt + 50, sd, k, j)))
                g = K.hull_gap(net, v, dirs)
                equal += g["equal"]
                tot += 1
                n_cuts.append(g["n_cuts"])
                if not g["equal"]:
                    gaps.append(g["max_gap_rel"])
        per_seed.append(equal / tot)
        n_total += tot
    mean, std = _mean_std(per_seed)
    return {"capmode": capmode, "runs": n_total, "equal_mean": mean, "equal_std": std, "gap_median": float(np.median(gaps)) if gaps else 0.0,
            "n_cuts_mean": float(np.mean(n_cuts)), "directions": n_dirs + 2 * max(ks)}


def range_classes(capmode: str, ks, runs_per_k: int, seeds=C.SWEEP_SEEDS, salt: int = 3) -> dict:
    """Spielraum im Kern je Kante nach Klasse (Veto, überzählig, übrige): Kern-Maximum und -Minimum relativ zu v(N), gemittelt über alle Kanten der Klasse."""
    acc = {"veto": [], "surplus": [], "other": []}
    for sd in seeds:
        for k in ks:
            for j in range(runs_per_k):
                net = S.make_net("Zufalls-DAG", k, capmode, instance_seed(salt, sd, k, j))
                v = N.coalition_values(net)
                if v[-1] <= 0:
                    continue
                veto, surplus = set(G.veto_edges(v)), set(G.surplus_edges(v))
                for i in range(net.m):
                    lo, hi = K.core_range(v, i)
                    key = "veto" if i in veto else ("surplus" if i in surplus else "other")
                    acc[key].append((lo / v[-1], hi / v[-1]))
    out = {}
    for key, rows in acc.items():
        a = np.array(rows) if rows else np.zeros((0, 2))
        out[key] = {"edges": len(rows), "max_mean": float(a[:, 1].mean()) if len(a) else None, "min_mean": float(a[:, 0].mean()) if len(a) else None,
                    "max_is_zero": int(np.sum(a[:, 1] < K.TOL)) if len(a) else 0}
    out["capmode"] = capmode
    return out


# ---------------------------------------------------------------------------------------------------- E3: Kapazitätsmonotonie
def monotonicity_edge(net: N.Net, i: int, caps=range(1, 9)) -> dict:
    """Shapley-Wert, Kern-Minimum und Kern-Maximum der Kante i, wenn nur ihre Kapazität die Werte in `caps` durchläuft."""
    phi, lo, hi = [], [], []
    for c in caps:
        v = N.coalition_values(net.with_capacity(i, c))
        phi.append(float(G.shapley(v)[i]))
        a, b = K.core_range(v, i)
        lo.append(a)
        hi.append(b)
    return {"caps": list(caps), "shapley": phi, "core_min": lo, "core_max": hi,
            "shapley_falls": bool(np.diff(phi).min() < -1e-9), "core_min_falls": bool(np.diff(lo).min() < -1e-7), "core_max_falls": bool(np.diff(hi).min() < -1e-7)}


def monotonicity_table(ks=(5, 6, 8), runs_per_k: int = 4, seeds=C.SWEEP_SEEDS, salt: int = 4) -> dict:
    pairs = phi_falls = lo_falls = hi_falls = 0
    example = None
    for sd in seeds:
        for k in ks:
            for j in range(runs_per_k):
                net = S.make_net("Zufalls-DAG", k, "1 bis 5", instance_seed(salt, sd, k, j))
                for i in range(net.m):
                    r = monotonicity_edge(net, i)
                    pairs += 1
                    phi_falls += r["shapley_falls"]
                    lo_falls += r["core_min_falls"]
                    hi_falls += r["core_max_falls"]
    return {"pairs": pairs, "shapley_falls": phi_falls, "core_min_falls": lo_falls, "core_max_falls": hi_falls, "ks": list(ks), "example": example}


# ------------------------------------------------------------------------------------------------------- Restfall, Konvexität
def restfall_table(ks=(3, 4, 5, 6, 7, 8), runs_per_seed: int = 25, seeds=C.SWEEP_SEEDS, salt: int = 5) -> dict:
    n = convex = convex_in_core = clean = clean_out = clean_out_nonconvex = surplus_and_core = 0
    for sd in seeds:
        for k in ks:
            for j in range(runs_per_seed):
                net = S.make_net("Zufalls-DAG", k, "1 bis 5", instance_seed(salt, sd, k, j))
                c = classify(net)
                n += 1
                convex += bool(c["convex"])
                convex_in_core += bool(c["convex"] and c["in_core"])
                surplus_and_core += bool(c["surplus"] and c["in_core"])
                if not c["surplus"]:
                    clean += 1
                    if not c["in_core"]:
                        clean_out += 1
                        clean_out_nonconvex += not c["convex"]
    return {"nets": n, "convex": convex, "convex_in_core": convex_in_core, "clean": clean, "clean_outside": clean_out,
            "clean_outside_nonconvex": clean_out_nonconvex, "surplus_and_in_core": surplus_and_core, "ks": list(ks)}


# --------------------------------------------------------------------------------------------------------- E4: Zeilenerzeugung
def rowgen_cell(m: int, runs: int, method: str, compare_full: bool = False, seed0: int = 0, salt: int = 6, time_limit: float = 20.0) -> dict:
    """Zeilenerzeugung für die größte Zahlung einer zufällig gewählten Kante eines minimalen Schnitts in Zufalls-DAG mit m Kanten (n = nodes_for bzw. BIG_NODES ab m = 21)."""
    n_nodes = C.BIG_NODES if m > 20 else None
    rows, rounds, equal, statuses, seconds, zero_target, targets = [], [], 0, [], [], 0, []
    for j in range(runs):
        net = S.make_random_net_k(m, "1 bis 5", instance_seed(salt, seed0, m, j), n=n_nodes)
        rng = SplitMix64(instance_seed(salt + 50, seed0, m, j))
        cut = N.min_cut_edges(net)
        i = cut[S.randint(rng, 0, len(cut) - 1)]       # eine Kante eines minimalen Schnitts: ihr Kern-Maximum ist positiv
        v = N.coalition_values(net) if (method == "enum" or compare_full) else None
        res = K.max_payment_by_row_generation(net, i, method, v=v, time_limit=time_limit)
        rows.append(res.rows)
        rounds.append(res.rounds)
        statuses.append(res.status)
        seconds.append(res.total_seconds)
        target = float(res.x[i]) if res.x is not None else float("nan")
        targets.append(target)
        zero_target += bool(abs(target) < K.TOL)
        if compare_full and m <= K.CORE_LP_MAX_PLAYERS:
            equal += abs(K.core_range(v, i)[1] - target) < 1e-6
    return {"m": m, "method": method, "runs": runs, "rows_mean": float(np.mean(rows)), "rows_min": int(min(rows)), "rows_max": int(max(rows)),
            "rounds_mean": float(np.mean(rounds)), "all_optimal": all(s == "optimal" for s in statuses), "equal_to_full": int(equal),
            "target_zero": int(zero_target), "coalitions": 1 << m, "seconds_mean": float(np.mean(seconds))}


def timed(func, *args, **kwargs) -> tuple:
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    return out, time.perf_counter() - t0
