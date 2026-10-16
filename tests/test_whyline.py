import json

import numpy as np
import pandas as pd
import pytest

import whyline as wl
from whyline.cli import main
from whyline.synth import Incident, make_funnel

CR = wl.Ratio("conversions", "sessions", name="conversion rate", percent=True)


def random_state(rng, k=40):
    d0 = rng.integers(50, 5000, k).astype(float)
    d1 = rng.integers(50, 5000, k).astype(float)
    n0 = np.floor(d0 * rng.uniform(0.01, 0.2, k))
    n1 = np.floor(d1 * rng.uniform(0.01, 0.2, k))
    return n0, d0, n1, d1


# --- maths -------------------------------------------------------------------

def test_ratio_components_are_exactly_additive_over_any_partition():
    rng = np.random.default_rng(0)
    n0, d0, n1, d1 = random_state(rng)
    total = CR.value(n1, d1) - CR.value(n0, d0)
    labels = rng.integers(0, 5, len(n0))
    parts = [sum(CR.components(n0, d0, n1, d1, labels == g).values()) for g in range(5)]
    assert sum(parts) == pytest.approx(total, abs=1e-14)


def test_whole_population_is_pure_rate():
    rng = np.random.default_rng(1)
    n0, d0, n1, d1 = random_state(rng)
    c = CR.components(n0, d0, n1, d1, np.ones(len(n0), bool))
    assert c["mix"] == pytest.approx(0, abs=1e-15)
    assert c["rate"] == pytest.approx(CR.value(n1, d1) - CR.value(n0, d0))


@pytest.mark.parametrize("mechanism", ["rate", "mix"])
def test_neutralize_restores_baseline_behaviour(mechanism):
    rng = np.random.default_rng(2)
    n0, d0, n1, d1 = random_state(rng)
    m = np.zeros(len(n0), bool)
    m[:7] = True
    nn, dd = CR.neutralize(n0, d0, n1, d1, m, mechanism)
    before = CR.segment_values(n0, d0, n1, d1, m)
    after = CR.segment_values(n0, d0, nn, dd, m)
    if mechanism == "rate":   # each cell gets its baseline rate, volumes untouched
        np.testing.assert_allclose(nn[m] / dd[m], n0[m] / d0[m])
        np.testing.assert_allclose(dd, d1)
    else:                     # segment back to baseline share, cell rates untouched
        assert after["current_share"] == pytest.approx(before["base_share"])
        np.testing.assert_allclose(nn[m] / dd[m], n1[m] / d1[m])
        np.testing.assert_allclose(dd[m] / dd[m].sum(), d0[m] / d0[m].sum())


def test_overlapping_neutralizations_commute():
    rng = np.random.default_rng(3)
    n0, d0, n1, d1 = random_state(rng)
    a = np.arange(len(n0)) < 20
    b = np.arange(len(n0)) % 3 == 0
    x = CR.neutralize(n0, d0, *CR.neutralize(n0, d0, n1, d1, a, "rate"), b, "mix")
    y = CR.neutralize(n0, d0, *CR.neutralize(n0, d0, n1, d1, b, "mix"), a, "rate")
    assert CR.value(*x) == pytest.approx(CR.value(*y))


def test_new_segment_is_pure_mix_and_finite():
    n0 = np.array([10.0, 0.0]); d0 = np.array([100.0, 0.0])
    n1 = np.array([10.0, 1.0]); d1 = np.array([100.0, 50.0])
    c = CR.components(n0, d0, n1, d1, np.array([False, True]))
    assert c["rate"] == 0 and np.isfinite(c["mix"]) and c["mix"] < 0


# --- end to end ----------------------------------------------------------------

def test_demo_recovers_every_planted_incident():
    ds = wl.synth.demo()
    r = wl.explain(ds.frame, CR, ds.dimensions, "week")
    found = {(tuple(sorted(c.segment.items())), c.mechanism) for c in r.causes}
    planted = {(tuple(sorted(i.segment.items())), i.mechanism) for i in ds.incidents}
    assert planted == found
    assert all(c.verdict == "strong" for c in r.causes)


def test_impacts_plus_residual_equal_observed_change():
    ds = wl.synth.demo(3)
    for metric in (CR, wl.Sum("revenue"), wl.Ratio("revenue", "conversions", name="AOV")):
        r = wl.explain(ds.frame, metric, ds.dimensions, "week")
        assert sum(c.impact for c in r.causes) + r.residual == pytest.approx(r.delta, rel=1e-9, abs=1e-12)


def test_sum_metric_finds_revenue_drop():
    ds = make_funnel(seed=4, incidents=[Incident({"country": "US"}, rate=0.5)])
    r = wl.explain(ds.frame, wl.Sum("revenue"), ds.dimensions, "week")
    assert r.causes[0].segment == {"country": "US"}
    assert r.causes[0].share > 0.8


def test_quiet_week_finds_nothing_material():
    ds = make_funnel(seed=5)
    r = wl.explain(ds.frame, CR, ds.dimensions, "week")
    assert len(r.causes) <= 1


def test_explicit_periods_and_errors():
    ds = wl.synth.demo()
    r = wl.explain(ds.frame, CR, ds.dimensions, "week", baseline="2026-W35", current="2026-W36")
    assert (r.baseline, r.current) == ("2026-W35", "2026-W36")
    with pytest.raises(KeyError):
        wl.explain(ds.frame, CR, ["nope"], "week")
    with pytest.raises(KeyError):
        wl.explain(ds.frame, CR, ds.dimensions, "week", baseline="1999-W01")


def test_outputs_render():
    ds = wl.synth.demo()
    r = wl.explain(ds.frame, CR, ds.dimensions, "week")
    assert "paid_social" in r.to_text()
    assert r.to_html().startswith("<!doctype html>")
    assert len(r.to_frame()) == len(r.causes)
    json.dumps(r.to_dict())


def test_cli(tmp_path, capsys):
    path = tmp_path / "data.csv"
    wl.synth.demo().frame.to_csv(path, index=False)
    html = tmp_path / "r.html"
    assert main(["explain", str(path), "--ratio", "conversions/sessions", "--by",
                 "country,device,channel,user_type", "--time", "week", "--json",
                 "--html", str(html)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {c["mechanism"] for c in out["causes"]} == {"rate", "mix"}
    assert html.read_text().startswith("<!doctype html>")
