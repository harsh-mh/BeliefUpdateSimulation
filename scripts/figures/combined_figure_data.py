"""
Collect the source data behind each panel (a-i) of combined_figure.

Each panel script writes a .csv next to the .svg it draws, from the same numbers
it plots. This module copies those files into one folder, named by panel letter.
A missing csv, or one older than its svg, raises an error, because the image
and the numbers might no longer match. The per-panel files are then stacked
into a single long-format csv for the whole figure.
"""

import os
import shutil

import pandas as pd

# panel letter -> (output name, source csv, panel svg it belongs to)
PANEL_SOURCES = {
    "a": ("panel_a_human_distribution",
          "figures/plots/distributions/human_distributions_nol"),
    "b": ("panel_b_comment_rankings",
          "figures/rankings/plot_avg_comment_rankings"),
    "c": ("panel_c_gpt52_distribution",
          "figures/plots/distributions/panel_gpt52"),
    "d": ("panel_d_gpt5mini_distribution",
          "figures/plots/distributions/panel_gpt5mini"),
    "e": ("panel_e_gemini_distribution",
          "figures/plots/distributions/panel_gemini"),
    "f": ("panel_f_claude_distribution",
          "figures/plots/distributions/panel_claude"),
    "g": ("panel_g_qwen_distribution",
          "figures/plots/distributions/panel_qwen"),
    "h": ("panel_h_llama_distribution",
          "figures/plots/distributions/panel_llama"),
    "i": ("panel_i_absolute_belief_change",
          "figures/plots/absolute_belief_change_dot_range_nol"),
}

# Script to rerun when a panel's csv is missing or stale.
PANEL_SCRIPTS = {
    "a": "scripts/figures/distributions/human_distributions.py",
    "b": "scripts/figures/inspect_rankings.py",
    "i": "scripts/figures/absolute_belief_change.py",
}
for _p in "cdefgh":
    PANEL_SCRIPTS[_p] = "scripts/figures/distributions/post_stance_models.py"

# Individual data points behind panels that plot summary statistics.
# key -> (output name, source csv, panel letter it belongs to)
POINT_SOURCES = {
    "i_points": ("panel_i_absolute_belief_change_points",
                 "figures/plots/absolute_belief_change_dot_range_nol_points", "i"),
}


def _sources():
    """key -> (output name, csv stem, svg stem, panel letter) for every csv to collect."""
    out = {letter: (name, stem, stem, letter)
           for letter, (name, stem) in PANEL_SOURCES.items()}
    for key, (name, stem, letter) in POINT_SOURCES.items():
        out[key] = (name, stem, PANEL_SOURCES[letter][1], letter)
    return out


def collect_panel_csvs(out_dir, root="."):
    """Copy each panel's csv(s) to out_dir/<name>.csv. Returns the written paths."""
    problems = []
    for name, csv_stem, svg_stem, letter in _sources().values():
        csv = os.path.join(root, f"{csv_stem}.csv")
        svg = os.path.join(root, f"{svg_stem}.svg")
        if not os.path.exists(csv):
            problems.append(f"({letter}) missing {csv_stem}.csv")
        elif os.path.exists(svg) and os.path.getmtime(csv) < os.path.getmtime(svg):
            problems.append(f"({letter}) {csv_stem}.csv is older than {svg_stem}.svg")
        else:
            continue
        problems[-1] += f" -> rerun {PANEL_SCRIPTS[letter]}"
    if problems:
        raise RuntimeError("Panel data out of date:\n  " + "\n  ".join(problems))

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, csv_stem, _, _ in _sources().values():
        dst = os.path.join(out_dir, f"{name}.csv")
        shutil.copyfile(os.path.join(root, f"{csv_stem}.csv"), dst)
        written.append(dst)
    return written


# Columns of the single full-figure csv. Cells that don't apply to a panel are empty.
FIGURE_COLUMNS = ["panel", "source", "persona_id", "topic", "package", "source_message_id",
                  "likert_value", "likert_label", "measure", "value"]


def _likert_rows(letter, df, value_cols):
    """value_cols: {csv column: (source, measure)}; source None = use 'model' column."""
    rows = []
    for _, r in df.iterrows():
        for col, (source, measure) in value_cols.items():
            rows.append({"panel": letter, "source": source or r["model"],
                         "likert_value": r["likert_value"],
                         "likert_label": r["likert_label"],
                         "measure": measure, "value": r[col]})
    return rows


def build_figure_table(panels):
    """Stack the per-panel tables {key: DataFrame} into one long table.

    Keys are panel letters, plus the POINT_SOURCES keys (e.g. "i_points") for
    the individual data points behind summary statistics.

    One row per number: which panel, which source (model or Human), where it
    sits (persona / topic / package / comment / Likert value), what it is
    (measure) and its value.
    """
    rows = []
    point_frames = []
    for letter, df in panels.items():
        if letter == "i_points":
            # One |delta| per (source, persona, topic): the values the
            # plotted mean and SD are computed over.
            long = df.rename(columns={"abs_delta": "value"}).assign(
                panel="i", measure="abs_delta")
            point_frames.append(long)
            continue
        if letter == "a":
            rows += _likert_rows(letter, df, {
                "human_post_stance_count": ("Human", "post_stance_count"),
                "human_initial_stance_count": ("Human", "initial_stance_count"),
            })
        elif letter in "cdefgh":
            rows += _likert_rows(letter, df, {
                "model_post_stance_count": (None, "post_stance_count"),
                "human_post_stance_count": ("Human", "post_stance_count"),
            })
        elif letter == "b":
            id_cols = ["topic", "package", "source_message_id"]
            for _, r in df.iterrows():
                for source in df.columns.drop(id_cols):
                    rows.append({"panel": letter, "source": source,
                                 **{c: r[c] for c in id_cols},
                                 "measure": "mean_rank", "value": r[source]})
        elif letter == "i":
            for _, r in df.iterrows():
                for measure in ["n", "mean_abs_delta", "sd_abs_delta",
                                "range_low", "range_high"]:
                    rows.append({"panel": letter, "source": r["source"],
                                 "topic": r["group"],
                                 "measure": measure, "value": r[measure]})
        else:
            raise ValueError(f"unknown panel {letter!r}")
    out = pd.concat([pd.DataFrame(rows, columns=FIGURE_COLUMNS)]
                    + [f.reindex(columns=FIGURE_COLUMNS) for f in point_frames],
                    ignore_index=True)
    for col in ["source_message_id", "likert_value"]:
        out[col] = out[col].astype("Int64")
    return out


def write_figure_csv(path, data_dir):
    """Combine the per-panel csvs in data_dir (from collect_panel_csvs) into one csv."""
    panels = {key: pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
              for key, (name, *_) in _sources().items()}
    table = build_figure_table(panels)
    table.to_csv(path, index=False)
    return table
