"""
Belief change (Δ = post − initial) distribution by topic, comparing models.
3 rows (topics: UBI, penalty, weight_loss) × 6 columns (one per model).
Each panel overlays human bars + a single LLM dot-line.

Run from: HumanSimulationProject/
Output:   figures/plots/delta_belief_by_topic_models.svg / .png / .pdf
          figures/plots/delta_belief_by_topic_models.csv (source data: the
          histogram behind every panel)
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS

# Journal style: the figure is drawn at its printed width (183 mm double
# column), so these sizes are the final printed sizes (allowed: 5-7 pt, Arial).
FIG_WIDTH_IN  = 7.2
FIG_HEIGHT_IN = 4.6
FS_TICK   = 5
FS_LABEL  = 6
FS_TITLE  = 6   # 7 pt bold is too wide for the longest model names

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "axes.linewidth":  0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  2,
    "ytick.major.size":  2,
})

with open("colors.yaml") as _f:
    _yaml = yaml.safe_load(_f)
_COLORS = _yaml["models"]

# ── model → (display label, data file, color) — from colors.yaml ─────────────
MODELS = {
    "GPT-5.2":                ("GPT-5.2",                "results/gpt5.2_results.xlsx",                                         _COLORS["GPT-5.2"]),
    "GPT-5-mini":             ("GPT-5-Mini",             "results/academic-ai-gpt-5-mini_all_personas_results.xlsx",            _COLORS["GPT-5-mini"]),
    "Claude Opus 4.6":        ("Claude-Opus-4.6",        "results/claude-opus-4-6_all_personas_results.xlsx",                  _COLORS["Claude Opus 4.6"]),
    "Gemini-3-flash":         ("Gemini-3-Flash-Preview", "results/gemini-3-flash-preview_all_personas_results.xlsx",           _COLORS["Gemini-3-flash"]),
    "Qwen3-32B":              ("Qwen3-32B",              "results/Qwen3_32B_all_personas_results.xlsx",                        _COLORS["Qwen3-32B"]),
    "Llama-3.3-70B-Instruct": ("Llama-3.3-70B-Instruct","results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",           _COLORS["Llama-3.3-70B-Instruct"]),
}

TOPIC_LABELS = {
    "UBI":         "UBI",
    "penalty":     "Penalty Shootouts",
    "weight_loss": "Weight Loss Drugs",
}

HUMAN_COLOR = _yaml["human"]
TOPICS      = ["UBI", "penalty", "weight_loss"]
DELTA_BINS  = np.arange(-4.5, 5.5, 1.0)
DELTA_TICKS = np.arange(-4, 5)


# ── data loading ──────────────────────────────────────────────────────────────

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "llm_new_belief"] *= -1
    df.loc[mask, "init_belief"]    *= -1
    return df


def load_human() -> pd.DataFrame:
    df = pd.read_csv("results/merged_llm_participants_data_with_normalized_beliefs.csv")
    df["delta"] = df["final_belief_normalized"] - df["initial_belief_normalized"]
    return df


def load_llm(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = normalize(df)
    df["delta"] = df["llm_new_belief"] - df["init_belief"]
    return df[["topic", "delta"]].dropna()


def histogram(delta: np.ndarray):
    """Counts and proportions of delta per integer value in DELTA_TICKS."""
    counts, _ = np.histogram(delta, bins=DELTA_BINS)
    total = counts.sum()
    props = counts / total if total > 0 else counts.astype(float)
    return counts, props


def delta_histogram_table(human_df, llm_cache):
    """The numbers plotted in every panel, one row per (topic, source, delta).

    Human rows appear once per topic (the same outline is drawn in every
    model's panel of that row). Model sources use the panel titles.
    """
    rows = []
    for topic in TOPICS:
        sources = [("Human", human_df)] + [(MODELS[k][0], llm_cache[k]) for k in MODELS]
        for source, df in sources:
            counts, props = histogram(df.loc[df["topic"] == topic, "delta"].values)
            for d, c, p in zip(DELTA_TICKS, counts, props):
                rows.append({"topic": topic, "topic_label": TOPIC_LABELS[topic],
                             "source": source, "delta": int(d),
                             "count": int(c), "proportion": p})
    return pd.DataFrame(rows)


# ── panel drawing ─────────────────────────────────────────────────────────────

def draw_panel(ax, h_props, l_props, color, show_ylabel, show_xlabel):
    # LLM: coloured bars (foreground)
    ax.bar(DELTA_TICKS, l_props, width=0.7, color=color, edgecolor="white",
           linewidth=0.3, zorder=2)

    # Human: grey step outline (background reference)
    edges = np.append(DELTA_TICKS - 0.5, DELTA_TICKS[-1] + 0.5)
    ax.stairs(h_props, edges, color=HUMAN_COLOR, lw=1.0, zorder=3, label="Human")

    ax.axvline(0, color="grey", lw=0.5, linestyle=":", zorder=0)
    ax.set_xticks(DELTA_TICKS)
    ax.set_xticklabels([str(v) for v in DELTA_TICKS])
    ax.set_xlim(-4.7, 4.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=FS_TICK, pad=1.5)

    if show_ylabel:
        ax.set_ylabel("Proportion", fontsize=FS_LABEL)
    if show_xlabel:
        ax.set_xlabel("Δ Belief (post − initial)", fontsize=FS_LABEL, labelpad=2)


# ── main ──────────────────────────────────────────────────────────────────────

def plot():
    human_df = load_human()

    model_keys  = list(MODELS.keys())
    n_models    = len(model_keys)

    # pre-load all LLM data
    llm_cache = {}
    for key in model_keys:
        _, path, _ = MODELS[key]
        llm_cache[key] = load_llm(path)

    # Every plotted number comes from this table, which is also saved as csv.
    table = delta_histogram_table(human_df, llm_cache)

    def props(topic, source):
        sel = table[(table["topic"] == topic) & (table["source"] == source)]
        return sel.sort_values("delta")["proportion"].values

    fig, axes = plt.subplots(
        3, n_models,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharex=True, sharey=False,
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.03, hspace=0.04)


    for row, topic in enumerate(TOPICS):
        for col, key in enumerate(model_keys):
            ax             = axes[row, col]
            label, _, color = MODELS[key]

            draw_panel(
                ax,
                h_props=props(topic, "Human"),
                l_props=props(topic, label),
                color=color,
                show_ylabel=(col == 0),
                show_xlabel=(row == 2),
            )

            if row == 0:
                ax.set_title(label, fontsize=FS_TITLE, fontweight="bold", pad=3, color=color)

            if col == 0:
                ax.set_ylabel(f"{TOPIC_LABELS[topic]}\n\nProportion", fontsize=FS_LABEL)

    # shared legend: grey line for Human, coloured patch per model
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    legend_handles = [
        Line2D([0], [0], color=HUMAN_COLOR, lw=1.0, label="Human"),
    ] + [
        mpatches.Patch(color=MODELS[k][2], label=MODELS[k][0])
        for k in model_keys
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside lower center", ncol=n_models + 1,
        fontsize=FS_LABEL, framealpha=0.9, frameon=False,
        handlelength=1.5, columnspacing=1.2,
    )

    # Saved at exactly FIG_WIDTH_IN x FIG_HEIGHT_IN (no bbox_inches="tight",
    # which would change the printed size and with it the font sizes).
    out = "figures/plots/delta_belief_by_topic_models"
    fig.savefig(f"{out}.svg", format="svg")
    fig.savefig(f"{out}.png", dpi=300)
    fig.savefig(f"{out}.pdf")
    table.to_csv(f"{out}.csv", index=False)
    print(f"Saved {out}.svg / .png / .pdf / .csv")


if __name__ == "__main__":
    plot()
