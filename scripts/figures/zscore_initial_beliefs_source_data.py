"""
Source data for zscore_initial_beliefs_combined (panels a-m) in one csv.

Panel a (z-scores) comes from figures/plots/zscore_initial_beliefs_panel_a_zscores.csv.
Those values were produced by an earlier version of the regression and are
supplied as-is; the current zscore_initial_beliefs_combined.py would compute
different numbers, so panel a is not recomputed here.

Panels b-m (initial / post-stance distributions of the own-initial-belief
ablation) are recomputed with the figure script's own loaders, counting each
Likert value the same way its draw_dist does.

Run from: repo root
Output:   figures/plots/zscore_initial_beliefs_combined_data.csv
"""

import pandas as pd

from scripts.figures.zscore_initial_beliefs_combined import (
    ABLATION_MODELS, DISPLAY_ROWS, LIKERT_LABELS, LIKERT_VALUES,
    load_ablation, load_human,
)

PANEL_A_CSV = "figures/plots/zscore_initial_beliefs_panel_a_zscores.csv"
OUT_CSV = "figures/plots/zscore_initial_beliefs_combined_data.csv"

COLUMNS = ["panel", "source", "predictor", "stance",
           "likert_value", "likert_label", "measure", "value"]


def panel_order():
    """[(letter, model, stance)] for panels b-m, in the figure's layout order."""
    out, letter = [], ord("b")
    for pair in DISPLAY_ROWS:
        for model in pair:
            for stance in ("initial", "post"):
                out.append((chr(letter), model, stance))
                letter += 1
    return out


def likert_counts(series):
    """Same counting as draw_dist in the figure script."""
    return [int((series == v).sum()) for v in LIKERT_VALUES]


def panel_a_rows(zscores):
    """Wide z-score table (Parameter x source) -> long rows for panel a."""
    long = zscores.melt(id_vars="Parameter", var_name="source", value_name="value")
    return long.rename(columns={"Parameter": "predictor"}).assign(
        panel="a", measure="z_value")[["panel", "source", "predictor", "measure", "value"]]


def distribution_rows(human_init, human_post, model_series):
    """Rows for panels b-m.

    model_series: {model label: (init series, post series)}.
    Each panel has the model's bars and the human outline for the same stance.
    """
    human = {"initial": human_init, "post": human_post}
    rows = []
    for letter, model, stance in panel_order():
        init, post = model_series[model]
        series = {"initial": init, "post": post}[stance]
        for source, s in ((model, series), ("Human", human[stance])):
            for v, lab, c in zip(LIKERT_VALUES, LIKERT_LABELS, likert_counts(s)):
                rows.append({"panel": letter, "source": source, "stance": stance,
                             "likert_value": v, "likert_label": lab,
                             "measure": "count", "value": c})
    return pd.DataFrame(rows)


def build_table(zscores, human_init, human_post, model_series):
    out = pd.concat([panel_a_rows(zscores),
                     distribution_rows(human_init, human_post, model_series)],
                    ignore_index=True).reindex(columns=COLUMNS)
    out["likert_value"] = out["likert_value"].astype("Int64")
    return out


def main():
    zscores = pd.read_csv(PANEL_A_CSV)
    human_init, human_post = load_human()
    model_series = {label: load_ablation(path) for label, path in ABLATION_MODELS}
    table = build_table(zscores, human_init, human_post, model_series)
    table.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV} ({len(table)} rows)")


if __name__ == "__main__":
    main()
