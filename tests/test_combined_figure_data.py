"""Per-panel source data behind combined_figure (panels a-i).

The figure modules read colors.yaml relative to the working directory, so these
tests assume pytest is run from the repository root.
"""
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from scripts.figures import absolute_belief_change as abc
from scripts.figures import inspect_rankings as ir
from scripts.figures.combined_figure_data import (
    FIGURE_COLUMNS, PANEL_SOURCES, POINT_SOURCES, build_figure_table, collect_panel_csvs,
    write_figure_csv,
)
from scripts.figures.distributions import human_distributions as hd
from scripts.figures.distributions import post_stance_models as psm


# -- panel a ---------------------------------------------------------------

def test_human_distribution_table_counts_each_likert_value():
    init = pd.Series([-2, -2, 0, 1, None, "x"])
    post = pd.Series([2, 2, 2, -1, 0])
    t = hd.human_distribution_table(init, post)
    assert list(t["likert_value"]) == [-2, -1, 0, 1, 2]
    assert list(t["likert_label"]) == ["SD", "D", "N", "A", "SA"]
    assert list(t["human_initial_stance_count"]) == [2, 0, 1, 1, 0]
    assert list(t["human_post_stance_count"]) == [0, 1, 1, 0, 3]


# -- panels c-h ------------------------------------------------------------

def test_model_panel_table_has_model_bars_and_human_outline():
    model = pd.Series([0, 0, 1, 2])
    human = pd.Series([-2, -1, -1])
    t = psm.model_panel_table(model, human)
    assert list(t["model_post_stance_count"]) == [0, 0, 2, 1, 1]
    assert list(t["human_post_stance_count"]) == [1, 2, 0, 0, 0]


def test_model_panel_csv_written_alongside_svg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("figures/plots/distributions")
    human = pd.Series([0, 1, 1])
    cache = {k: pd.Series([-2, 2]) for k in psm.MODELS}
    order = list(psm.MODELS)
    psm.save_individual_panels(cache, human, order)
    t = pd.read_csv("figures/plots/distributions/panel_claude.csv")
    assert set(t["model"]) == {"Claude Opus 4.6"}
    assert list(t["model_post_stance_count"]) == [1, 0, 0, 0, 1]
    assert list(t["human_post_stance_count"]) == [0, 0, 1, 2, 0]


# -- panel b ---------------------------------------------------------------

def _ranking_df():
    slugs = [slug for _, slug, _ in ir.MODELS.values()] + ["human"]
    rows = []
    # two personas rate the same (topic, package); ranks differ so means are non-trivial
    for persona, ranks in [("p1", (1, 2, 3)), ("p2", (2, 1, 3))]:
        row = {"persona_id": persona, "topic": "UBI", "package": "A"}
        for s in slugs:
            for i, r in zip((1, 2, 3), ranks):
                row[f"message_{i}_{s}_rank"] = r
        rows.append(row)
    return pd.DataFrame(rows)


def test_avg_comment_rank_table_one_row_per_comment_one_column_per_source():
    t = ir.avg_comment_rank_table(_ranking_df())
    sources = list(ir.MODELS) + ["Human"]
    assert list(t.columns) == ["topic", "package", "source_message_id"] + sources
    assert list(t["source_message_id"]) == [1, 2, 3]
    for s in sources:
        assert list(t[s]) == [1.5, 1.5, 3.0]


# -- panel i ---------------------------------------------------------------

def test_dot_range_table_matches_compute_stats():
    df = pd.DataFrame({
        "topic": ["UBI", "UBI", "penalty", "weight_loss"],
        "abs_delta": [0.0, 1.0, 2.0, 1.0],
    })
    stats = abc.compute_stats(df)
    t = abc.dot_range_table([("Human", "#000", stats)])
    row = t.iloc[0]
    assert row["source"] == "Human"
    assert row["n"] == 4
    assert row["mean_abs_delta"] == pytest.approx(1.0)
    sd = np.std([0, 1, 2, 1], ddof=1)
    assert row["sd_abs_delta"] == pytest.approx(sd)
    assert row["range_low"] == pytest.approx(1.0 - sd)
    assert row["range_high"] == pytest.approx(1.0 + sd)


def _points_frame(persona_ids, topics, init, new):
    df = pd.DataFrame({"persona_id": persona_ids, "topic": topics,
                       "initial_belief": init, "new_belief": new})
    df["abs_delta"] = (df["new_belief"] - df["initial_belief"]).abs()
    return df


def test_dot_range_points_reproduce_plotted_mean_and_sd():
    human = _points_frame(["p1", "p2", "p3"], ["UBI", "UBI", "penalty"], [0, 1, -2], [2, 1, 0])
    model = _points_frame(["p1", "p2"], ["UBI", "penalty"], [0, 0], [-1, 0])
    pts = abc.dot_range_points([("Human", human), ("GPT-5.2", model)])
    assert list(pts.columns) == ["source", "persona_id", "topic", "abs_delta"]
    assert list(pts["source"]) == ["Human"] * 3 + ["GPT-5.2"] * 2

    summary = abc.dot_range_table(
        [("Human", "#000", abc.compute_stats(human)),
         ("GPT-5.2", "#111", abc.compute_stats(model))]).set_index("source")
    for source, grp in pts.groupby("source"):
        assert len(grp) == summary.loc[source, "n"]
        assert grp["abs_delta"].mean() == pytest.approx(summary.loc[source, "mean_abs_delta"])
        assert grp["abs_delta"].std() == pytest.approx(summary.loc[source, "sd_abs_delta"])


def test_load_human_uses_normalized_beliefs_not_raw(monkeypatch):
    raw = pd.DataFrame({
        "persona_id": ["p1", "p2"],
        "topic": ["UBI", "penalty"],
        "initial_belief": [9, 9],             # raw columns must be ignored
        "final_belief": [9, 9],
        "initial_belief_normalized": [1, -2],
        "final_belief_normalized": [-1, -2],
    })
    monkeypatch.setattr(abc.pd, "read_csv", lambda path: raw.copy())
    out = abc.load_human()
    assert list(out.columns) == abc.POINT_COLUMNS
    assert list(out["initial_belief"]) == [1, -2]
    assert list(out["new_belief"]) == [-1, -2]
    assert list(out["abs_delta"]) == [2, 0]


def test_load_llm_keeps_ids_and_flips_negative_wording(monkeypatch):
    from belief_update_sim.normalization import NEGATIVE_FORMULATIONS
    neg = sorted(NEGATIVE_FORMULATIONS)[0]
    raw = pd.DataFrame({
        "persona_id": ["p1", "p2", "p3"],
        "topic": ["UBI", "UBI", "penalty"],
        "statement_formulation": ["positive wording", neg, "positive wording"],
        "init_belief": [1, 1, 0],
        "llm_new_belief": [2, -1, None],
    })
    monkeypatch.setattr(abc.pd, "read_excel", lambda path: raw.copy())
    out = abc.load_llm("unused.xlsx")
    assert list(out.columns) == abc.POINT_COLUMNS
    assert list(out["persona_id"]) == ["p1", "p2"]       # missing answer dropped
    assert list(out["initial_belief"]) == [1, -1]         # p2 flipped
    assert list(out["new_belief"]) == [2, 1]
    assert list(out["abs_delta"]) == [1, 2]


# -- collecting into the combined figure folder ------------------------------

def _write_sources(root, csv_after_svg=True):
    stems = [stem for _, stem in PANEL_SOURCES.values()]
    for stem in stems:
        (root / stem).parent.mkdir(parents=True, exist_ok=True)
        (root / f"{stem}.svg").write_text("<svg/>")
    for stem in stems + [stem for _, stem, _ in POINT_SOURCES.values()]:
        (root / f"{stem}.csv").write_text(f"x\n{stem}\n")
        if not csv_after_svg:
            os.utime(root / f"{stem}.csv", (1, 1))


def test_collect_panel_csvs_copies_all_panels_and_points(tmp_path):
    _write_sources(tmp_path)
    out = tmp_path / "out"
    written = collect_panel_csvs(str(out), root=str(tmp_path))
    assert len(written) == 10
    names = sorted(p.name for p in out.iterdir())
    assert sorted({n[6] for n in names}) == list("abcdefghi")
    assert "panel_i_absolute_belief_change_points.csv" in names
    a = pd.read_csv(out / "panel_a_human_distribution.csv")
    assert a["x"][0] == PANEL_SOURCES["a"][1]


def test_collect_panel_csvs_rejects_missing_points_csv(tmp_path):
    _write_sources(tmp_path)
    os.remove(tmp_path / f"{POINT_SOURCES['i_points'][1]}.csv")
    with pytest.raises(RuntimeError, match=r"\(i\) missing.*_points.*absolute_belief_change"):
        collect_panel_csvs(str(tmp_path / "out"), root=str(tmp_path))


def test_collect_panel_csvs_rejects_missing_csv(tmp_path):
    _write_sources(tmp_path)
    os.remove(tmp_path / f"{PANEL_SOURCES['b'][1]}.csv")
    with pytest.raises(RuntimeError, match=r"\(b\) missing.*inspect_rankings"):
        collect_panel_csvs(str(tmp_path / "out"), root=str(tmp_path))
    assert not (tmp_path / "out").exists()


def test_collect_panel_csvs_rejects_csv_older_than_svg(tmp_path):
    _write_sources(tmp_path, csv_after_svg=False)
    with pytest.raises(RuntimeError, match="older than"):
        collect_panel_csvs(str(tmp_path / "out"), root=str(tmp_path))


# -- single csv for the full figure ------------------------------------------

def _panel_frames():
    """Small per-panel tables in the same shape the panel scripts write."""
    likert = [-2, -1, 0, 1, 2]
    labels = ["SD", "D", "N", "A", "SA"]
    frames = {"a": hd.human_distribution_table(pd.Series([0, 1]), pd.Series([2, 2, -2]))}
    frames["b"] = ir.avg_comment_rank_table(_ranking_df())
    for letter, model in zip("cdefgh", psm.MODELS):
        t = psm.model_panel_table(pd.Series([-1]), pd.Series([2, 2, -2]))
        t.insert(0, "model", model)
        frames[letter] = t
    df = _points_frame(["p1", "p2"], ["UBI", "penalty"], [0, 0], [0, 2])
    stats = abc.compute_stats(df)
    frames["i"] = abc.dot_range_table([("Human", "#000", stats), ("GPT-5.2", "#111", stats)])
    frames["i_points"] = abc.dot_range_points([("Human", df), ("GPT-5.2", df)])
    assert list(frames["a"]["likert_value"]) == likert
    assert list(frames["a"]["likert_label"]) == labels
    return frames


def test_figure_table_has_one_row_per_plotted_number():
    t = build_figure_table(_panel_frames())
    assert list(t.columns) == FIGURE_COLUMNS
    counts = t["panel"].value_counts().to_dict()
    assert counts["a"] == 10            # 5 Likert values x (post, initial)
    assert counts["b"] == 3 * 7         # 3 comments x 7 sources
    for letter in "cdefgh":
        assert counts[letter] == 10     # 5 Likert values x (model, human)
    # 2 sources x 5 summary measures + 2 sources x 2 responses (one |delta| each)
    assert counts["i"] == 2 * 5 + 2 * 2
    assert sorted(counts) == list("abcdefghi")


def test_figure_table_values_match_panel_tables():
    frames = _panel_frames()
    t = build_figure_table(frames)

    a = t[(t.panel == "a") & (t.measure == "initial_stance_count")]
    assert list(a["value"]) == list(frames["a"]["human_initial_stance_count"])
    assert set(a["source"]) == {"Human"}

    c = t[t.panel == "c"]
    model = c[c.source == "GPT-5.2"]
    assert list(model["likert_label"]) == ["SD", "D", "N", "A", "SA"]
    assert list(model["value"]) == [0, 1, 0, 0, 0]
    assert list(c[c.source == "Human"]["value"]) == [1, 0, 0, 0, 2]

    b = t[(t.panel == "b") & (t.source == "Human")]
    assert list(b["source_message_id"]) == [1, 2, 3]
    assert list(b["value"]) == [1.5, 1.5, 3.0]
    assert set(b["topic"]) == {"UBI"}

    i_all = t[(t.panel == "i") & (t.source == "GPT-5.2")]
    i = i_all[i_all.persona_id.isna()].set_index("measure")["value"]
    assert i["mean_abs_delta"] == pytest.approx(1.0)
    assert i["n"] == 2
    assert set(i_all[i_all.persona_id.isna()]["topic"]) == {"overall"}

    # individual points: one |delta| per response; they reproduce the summary rows
    deltas = i_all[i_all.persona_id.notna()]
    assert set(deltas["measure"]) == {"abs_delta"}
    assert list(deltas["persona_id"]) == ["p1", "p2"]
    assert list(deltas["topic"]) == ["UBI", "penalty"]
    assert list(deltas["value"]) == [0, 2]
    assert deltas["value"].mean() == pytest.approx(i["mean_abs_delta"])
    assert deltas["value"].std() == pytest.approx(i["sd_abs_delta"])


def test_write_figure_csv_round_trips(tmp_path):
    names = {**{k: v[0] for k, v in PANEL_SOURCES.items()},
             **{k: v[0] for k, v in POINT_SOURCES.items()}}
    for key, frame in _panel_frames().items():
        frame.to_csv(tmp_path / f"{names[key]}.csv", index=False)
    out = tmp_path / "figure.csv"
    written = write_figure_csv(str(out), str(tmp_path))
    back = pd.read_csv(out)
    assert list(back.columns) == FIGURE_COLUMNS
    assert len(back) == len(written)
