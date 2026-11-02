"""Source data behind the delta_belief_by_topic_models figure.

The figure module reads colors.yaml relative to the working directory, so these
tests assume pytest is run from the repository root.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from scripts.figures import delta_belief_by_topic_models as dbt


def _frames():
    human = pd.DataFrame({"topic": ["UBI", "UBI", "UBI", "penalty"],
                          "delta": [0, 0, -4, 2]})
    llm = {k: pd.DataFrame({"topic": ["UBI", "weight_loss"], "delta": [1, 4]})
           for k in dbt.MODELS}
    return human, llm


def test_histogram_counts_each_integer_delta():
    counts, props = dbt.histogram(np.array([-4, 0, 0, 4, 4, 4]))
    assert list(counts) == [1, 0, 0, 0, 2, 0, 0, 0, 3]
    assert props.sum() == pytest.approx(1.0)
    assert props[-1] == pytest.approx(0.5)


def test_histogram_of_empty_topic_is_all_zero():
    counts, props = dbt.histogram(np.array([]))
    assert counts.sum() == 0 and props.sum() == 0


def test_table_has_one_row_per_topic_source_delta():
    t = dbt.delta_histogram_table(*_frames())
    assert list(t.columns) == ["topic", "topic_label", "source", "delta", "count", "proportion"]
    n_sources = 1 + len(dbt.MODELS)
    assert len(t) == len(dbt.TOPICS) * n_sources * 9
    assert list(pd.unique(t["source"])) == ["Human"] + [v[0] for v in dbt.MODELS.values()]


def test_table_values():
    t = dbt.delta_histogram_table(*_frames())
    h = t[(t.topic == "UBI") & (t.source == "Human")].set_index("delta")
    assert h.loc[0, "count"] == 2 and h.loc[-4, "count"] == 1
    assert h.loc[0, "proportion"] == pytest.approx(2 / 3)
    assert h["count"].sum() == 3

    m = t[(t.topic == "weight_loss") & (t.source == "GPT-5-Mini")].set_index("delta")
    assert m.loc[4, "count"] == 1 and m.loc[4, "proportion"] == 1.0

    # topic with no responses for a source: zeros, not missing rows
    hw = t[(t.topic == "weight_loss") & (t.source == "Human")]
    assert len(hw) == 9 and hw["count"].sum() == 0


def test_plot_draws_from_table_and_writes_csv(tmp_path, monkeypatch):
    human, llm = _frames()
    monkeypatch.setattr(dbt, "load_human", lambda: human)
    monkeypatch.setattr(dbt, "load_llm", lambda path: next(iter(llm.values())))
    (tmp_path / "figures" / "plots").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    dbt.plot()
    t = pd.read_csv("figures/plots/delta_belief_by_topic_models.csv")
    pd.testing.assert_frame_equal(t, dbt.delta_histogram_table(human, llm),
                                  check_dtype=False)
    for ext in ("svg", "png", "pdf"):
        assert (tmp_path / f"figures/plots/delta_belief_by_topic_models.{ext}").stat().st_size > 0
