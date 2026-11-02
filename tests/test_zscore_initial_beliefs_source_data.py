"""Source data for the zscore_initial_beliefs_combined figure (panels a-m).

The figure module reads colors.yaml relative to the working directory, so these
tests assume pytest is run from the repository root.
"""
import matplotlib
matplotlib.use("Agg")

import pandas as pd

from scripts.figures import zscore_initial_beliefs_source_data as sd


def _zscores():
    return pd.DataFrame({"Parameter": ["Familiarity", "Initial Belief"],
                         "Human": [2.5, -14.0], "GPT-5.2": [1.0, -20.0]})


def _models():
    models = [m for pair in sd.DISPLAY_ROWS for m in pair]
    return {m: (pd.Series([0, 0, 1]), pd.Series([-2, 2])) for m in models}


def test_panel_order_matches_figure_layout():
    order = sd.panel_order()
    assert [p[0] for p in order] == list("bcdefghijklm")
    assert order[0] == ("b", "GPT-5.2", "initial")
    assert order[1] == ("c", "GPT-5.2", "post")
    assert order[3] == ("e", "GPT-5-mini", "post")
    assert order[-1] == ("m", "Llama-3.3-70B-Instruct", "post")


def test_panel_a_rows_keep_supplied_values_unchanged():
    rows = sd.panel_a_rows(_zscores())
    assert len(rows) == 4
    v = rows.set_index(["predictor", "source"])["value"]
    assert v[("Initial Belief", "Human")] == -14.0
    assert v[("Familiarity", "GPT-5.2")] == 1.0
    assert set(rows["measure"]) == {"z_value"}


def test_distribution_rows_model_bars_and_human_outline():
    human_init, human_post = pd.Series([-1, -1, 0]), pd.Series([1, 2, 2, 2])
    rows = sd.distribution_rows(human_init, human_post, _models())
    assert len(rows) == 12 * 2 * 5

    b = rows[rows.panel == "b"]
    assert set(b["stance"]) == {"initial"}
    assert list(b[b.source == "GPT-5.2"]["value"]) == [0, 0, 2, 1, 0]
    assert list(b[b.source == "Human"]["value"]) == [0, 2, 1, 0, 0]

    c = rows[rows.panel == "c"]
    assert set(c["stance"]) == {"post"}
    assert list(c[c.source == "GPT-5.2"]["value"]) == [1, 0, 0, 0, 1]
    assert list(c[c.source == "Human"]["value"]) == [0, 0, 0, 1, 3]


def test_build_table_single_long_table():
    t = sd.build_table(_zscores(), pd.Series([0]), pd.Series([0]), _models())
    assert list(t.columns) == sd.COLUMNS
    assert sorted(t["panel"].unique()) == list("abcdefghijklm")
    assert t[t.panel == "a"]["likert_value"].isna().all()
    assert t[t.panel != "a"]["predictor"].isna().all()


def test_supplied_panel_a_file_has_all_predictors_and_sources():
    z = pd.read_csv(sd.PANEL_A_CSV)
    assert len(z) == 12
    assert list(z.columns[1:]) == ["Human", "Llama-3.3-70B-Instruct", "Qwen3-32B", "GPT-5.2",
                                   "GPT-5-mini", "Gemini-3-flash-preview", "Claude-opus-4-6"]
    assert z.set_index("Parameter").loc["Initial Belief", "Human"] == -14.6677
