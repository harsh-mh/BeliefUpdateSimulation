"""
Mean and SD of absolute belief change (|Δ|) for humans and 6 LLMs, by topic.
Groups: Penalty Shootouts, UBI, Weight Loss Drugs, Overall.
Run from: HumanSimulationProject/
Output: figures/plots/absolute_belief_change_mean.png / .pdf / .eps
          figures/plots/absolute_belief_change_std.png / .pdf / .eps
"""
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import os
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS

# Set Nature Communications style parameters with consistent font sizes
plt.rcParams.update({
    'font.size': 12.5,  # Increased from 10 to 12.5 (+25%)
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.labelsize': 12.5,  # Increased from 10 to 12.5 (+25%)
    'axes.titlesize': 13.75,  # Increased from 11 to 13.75 (+25%)
    'xtick.labelsize': 11.25,  # Increased from 9 to 11.25 (+25%)
    'ytick.labelsize': 11.25,  # Increased from 9 to 11.25 (+25%)
    'legend.fontsize': 11.25,  # Increased from 9 to 11.25 (+25%)
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 0.8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5
})


with open("colors.yaml") as _f:
    _yaml = yaml.safe_load(_f)
_COLORS = _yaml["models"]
HUMAN_COLOR = _yaml["human"]

# Updated model order: Human first, then GPT-5.2, GPT-5-mini, Gemini, Claude, Qwen, Llama
MODELS_ORDERED = [
    ("Human", None, HUMAN_COLOR),  # Human placeholder
    ("GPT-5.2", "results/gpt5.2_results.xlsx", _COLORS["GPT-5.2"]),
    ("GPT-5-mini", "results/academic-ai-gpt-5-mini_all_personas_results.xlsx", _COLORS["GPT-5-mini"]),
    ("Gemini-3-flash", "results/gemini-3-flash-preview_all_personas_results.xlsx", _COLORS["Gemini-3-flash"]),
    ("Claude Opus 4.6", "results/claude-opus-4-6_all_personas_results.xlsx", _COLORS["Claude Opus 4.6"]),
    ("Qwen3-32B", "results/Qwen3_32B_all_personas_results.xlsx", _COLORS["Qwen3-32B"]),
    ("Llama-3.3-70B-Instruct", "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx", _COLORS["Llama-3.3-70B-Instruct"]),
]

GROUPS = ["penalty", "UBI", "weight_loss", "overall"]
GROUP_LABELS = ["Penalty Shootouts", "Universal Basic Income", "Weight Loss Drugs", "Overall"]

# Create output directory
os.makedirs("figures/plots", exist_ok=True)

def normalize_llm(df):
    df = df.copy()
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "llm_new_belief"] *= -1
    df.loc[mask, "init_belief"] *= -1
    return df

POINT_COLUMNS = ["persona_id", "topic", "initial_belief", "new_belief", "abs_delta"]


def load_human():
    """One row per (persona, topic): normalized initial / final belief and |delta|."""
    df = pd.read_csv("results/merged_llm_participants_data_with_normalized_beliefs.csv")
    # The csv also holds the raw (un-normalized) initial_belief / final_belief,
    # so pick the normalized columns before renaming them.
    df = df[["persona_id", "topic", "initial_belief_normalized", "final_belief_normalized"]]
    df = df.rename(columns={"initial_belief_normalized": "initial_belief",
                            "final_belief_normalized": "new_belief"})
    df["abs_delta"] = (df["new_belief"] - df["initial_belief"]).abs()
    return df[POINT_COLUMNS].dropna(subset=["topic", "abs_delta"])

def load_llm(path):
    """One row per (persona, topic): normalized initial / new belief and |delta|."""
    df = pd.read_excel(path)
    df = normalize_llm(df)
    df = df.rename(columns={"init_belief": "initial_belief", "llm_new_belief": "new_belief"})
    df["abs_delta"] = (df["new_belief"] - df["initial_belief"]).abs()
    return df[POINT_COLUMNS].dropna(subset=["topic", "abs_delta"])


def dot_range_points(frames):
    """Every |delta| behind the dot-range panel's means and SDs.

    One row per (source, persona, topic) response.
    frames: list of (source label, DataFrame from load_human / load_llm).
    """
    cols = ["persona_id", "topic", "abs_delta"]
    return pd.concat(
        [df[cols].assign(source=label) for label, df in frames],
        ignore_index=True,
    )[["source"] + cols]

def compute_stats(df):
    """Returns {group: (mean, sd, n)} of |delta| per topic and overall."""
    stats = {}
    for topic in ["penalty", "UBI", "weight_loss"]:
        sub = df.loc[df["topic"] == topic, "abs_delta"]
        stats[topic] = (sub.mean(), sub.std(), len(sub))
    stats["overall"] = (df["abs_delta"].mean(), df["abs_delta"].std(), len(df))
    return stats


def dot_range_table(all_stats, group="overall"):
    """The numbers plotted in the dot-range panel: mean dot and mean ± SD bar."""
    rows = []
    for label, _, stats in all_stats:
        mean, sd, n = stats[group]
        rows.append({
            "source": label,
            "group": group,
            "n": n,
            "mean_abs_delta": mean,
            "sd_abs_delta": sd,
            "range_low": mean - sd,
            "range_high": mean + sd,
        })
    return pd.DataFrame(rows)

def compute_percentile_stats(df, lo=25, hi=75):
    """Returns {topic: (median, lo_pct, hi_pct)} for each group."""
    stats = {}
    for topic in ["penalty", "UBI", "weight_loss"]:
        sub = df.loc[df["topic"] == topic, "abs_delta"]
        stats[topic] = (np.median(sub), np.percentile(sub, lo), np.percentile(sub, hi))
    sub = df["abs_delta"]
    stats["overall"] = (np.median(sub), np.percentile(sub, lo), np.percentile(sub, hi))
    return stats

def save_figure(fig, out_stem):
    """Save figure in multiple formats required for Nature Communications."""
    for ext in ['.png', '.pdf', '.eps', '.svg']:
        if ext == '.svg':
            fig.savefig(f"{out_stem}{ext}", format='svg', bbox_inches='tight')
        else:
            fig.savefig(f"{out_stem}{ext}", dpi=300, bbox_inches='tight')
    print(f"Saved {out_stem}.png / .pdf / .eps / .svg")

def draw_chart_with_errorbars(all_stats, out_stem):
    """Bar chart with error bars - optimized for two-column layout."""
    n_entities = len(all_stats)
    n_groups = len(GROUPS)
    bar_w = 0.8 / n_entities
    x = np.arange(n_groups)
    
    # Single column width for Nature Communications (8.5cm height, 8.5cm width typical)
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    
    for i, (label, color, stats) in enumerate(all_stats):
        offset = (i - (n_entities - 1) / 2) * bar_w
        means = [stats[g][0] for g in GROUPS]
        stds = [stats[g][1] for g in GROUPS]
        bars = ax.bar(x + offset, means, width=bar_w * 0.90,
                        color=color, edgecolor="white", linewidth=0.3,
                        zorder=2)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="black", linewidth=0.5,
                    capsize=1.5, capthick=0.5, zorder=3)
        
        # Add value labels ABOVE the bars with increased font size
        for bar, v in zip(bars, means):
            label_y = bar.get_height() + 0.01
            ax.text(bar.get_x() + bar.get_width() / 2,
                    label_y, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=11.25, fontweight="normal")

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, fontsize=11.25, rotation=15, ha='right')  # Increased from 9
    ax.set_ylabel("Mean |Δ|", fontsize=14)  # Increased from 10
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    max_value = max(
        stats[g][0] + stats[g][1]
        for _, _, stats in all_stats for g in GROUPS
    )
    ax.set_ylim(0, max_value * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11.25)  # Increased from 9

    handles = [mpatches.Patch(color=color, label=label, linewidth=0)
               for label, color, _ in all_stats]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=11.25, framealpha=0.9, bbox_to_anchor=(0.5, -0.12))  # Increased from 9
    
    save_figure(fig, out_stem)
    plt.close()

def draw_line_with_bands(all_stats, out_stem):
    """Line plot with shaded bands - Nature Communications style."""
    x = np.arange(len(GROUPS))
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    
    for label, color, stats in all_stats:
        means = np.array([stats[g][0] for g in GROUPS])
        stds = np.array([stats[g][1] for g in GROUPS])
        lw = 1.2 if label == "Human" else 0.8
        alpha = 0.15 if label == "Human" else 0.08
        ax.plot(x, means, color=color, linewidth=lw, marker="o",
                markersize=4, zorder=3, label=label)  # Increased markersize from 3 to 4
        ax.fill_between(x, means - stds, means + stds,
                        color=color, alpha=alpha, zorder=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, fontsize=11.25, rotation=15, ha='right')  # Increased from 9
    ax.set_ylabel("Mean |Δ|", fontsize=14)  # Increased from 10
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.set_ylim(0, max(
        stats[g][0] + stats[g][1]
        for _, _, stats in all_stats for g in GROUPS
    ) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11.25)  # Increased from 9

    fig.legend(fontsize=11.25, framealpha=0.9, loc="lower center",  # Increased from 9
               bbox_to_anchor=(0.5, -0.12), ncol=4)
    
    save_figure(fig, out_stem)
    plt.close()

def draw_dot_range(all_stats, out_stem):
    """Cleveland dot plot – Overall only, single panel sized for 2-col display."""
    n_entities = len(all_stats)
    group       = "overall"
    group_label = "Overall"

    # figsize matches 2-col display width (FIG_W/3 * 2 ≈ 9.3") in combined figure
    fig, ax = plt.subplots(1, 1, figsize=(9.3, 2.8), constrained_layout=True)

    y_positions = np.arange(n_entities) * 0.4

    table = dot_range_table(all_stats, group)
    x_min = table["range_low"].min()
    x_max = table["range_high"].max()
    x_pad = (x_max - x_min) * 0.05
    x_min -= x_pad
    x_max += x_pad

    for i, ((label, color, _), row) in enumerate(zip(all_stats, table.itertuples())):
        mean, std = row.mean_abs_delta, row.sd_abs_delta
        y = y_positions[i]
        ax.plot([mean - std, mean + std], [y, y],
                color=color, linewidth=3, solid_capstyle="round", zorder=2)
        ax.scatter(mean, y, color="white", s=30, zorder=4, linewidths=0)  # Increased from 25
        ax.scatter(mean, y, color=color, s=18, zorder=5, edgecolors="white",  # Increased from 15
                   linewidths=0.3)
        ax.text(mean, y + 0.09, f"{mean:.2f}",
                ha="center", va="bottom", fontsize=13, color=color)

    ax.axvline(0, color="#cccccc", lw=0.5, zorder=1)
    ax.set_title(group_label, fontsize=13.75, fontweight="normal")  # Increased from 10
    ax.set_xlabel("|Δ|", fontsize=14)  # Increased from 10
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False, labelsize=11.25)  # Increased from 9
    ax.set_xlim(x_min, x_max)
    pad = 0.2
    ax.set_ylim(y_positions[0] - pad, y_positions[-1] + pad)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for label, _, _ in all_stats], fontsize=13)
    ax.grid(axis='x', alpha=0.15, linestyle='-', linewidth=0.3)

    # Save clean version (no legend) for combined figure
    for ext in ['.png', '.pdf', '.svg']:
        if ext == '.svg':
            fig.savefig(f"{out_stem}_nol{ext}", format='svg', bbox_inches='tight')
        else:
            fig.savefig(f"{out_stem}_nol{ext}", dpi=300, bbox_inches='tight')
    table.to_csv(f"{out_stem}_nol.csv", index=False)

    handles = [mpatches.Patch(color=color, label=label, linewidth=0)
               for label, color, _ in all_stats]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=11.25, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))  # Increased from 9

    save_figure(fig, out_stem)
    plt.close()

def draw_dot_range_percentile(all_stats, out_stem, lo=25, hi=75):
    """Cleveland dot plot using median ± percentile range instead of mean ± SD."""
    n_entities = len(all_stats)
    group       = "overall"
    group_label = "Overall"

    fig, ax = plt.subplots(1, 1, figsize=(9.3, 2.8), constrained_layout=True)

    y_positions = np.arange(n_entities) * 0.4

    raw_vals = [(s[group][1], s[group][2]) for _, _, s in all_stats]
    x_min = min(lo_v for lo_v, _ in raw_vals)
    x_max = max(hi_v for _, hi_v in raw_vals)
    x_pad = (x_max - x_min) * 0.05
    x_min -= x_pad
    x_max += x_pad

    for i, (label, color, stats) in enumerate(all_stats):
        median, lo_v, hi_v = stats[group]
        y = y_positions[i]
        ax.plot([lo_v, hi_v], [y, y],
                color=color, linewidth=3, solid_capstyle="round", zorder=2)
        ax.scatter(median, y, color="white", s=30, zorder=4, linewidths=0)
        ax.scatter(median, y, color=color, s=18, zorder=5, edgecolors="white",
                   linewidths=0.3)
        ax.text(median, y + 0.09, f"{median:.2f}",
                ha="center", va="bottom", fontsize=13, color=color)

    ax.axvline(0, color="#cccccc", lw=0.5, zorder=1)
    ax.set_title(group_label, fontsize=13.75, fontweight="normal")
    ax.set_xlabel(f"|Δ| (median, {lo}th–{hi}th pct)", fontsize=14)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False, labelsize=11.25)
    ax.set_xlim(x_min, x_max)
    pad = 0.2
    ax.set_ylim(y_positions[0] - pad, y_positions[-1] + pad)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for label, _, _ in all_stats], fontsize=13)
    ax.grid(axis='x', alpha=0.15, linestyle='-', linewidth=0.3)

    for ext in ['.png', '.pdf', '.svg']:
        if ext == '.svg':
            fig.savefig(f"{out_stem}_nol{ext}", format='svg', bbox_inches='tight')
        else:
            fig.savefig(f"{out_stem}_nol{ext}", dpi=300, bbox_inches='tight')

    handles = [mpatches.Patch(color=color, label=label, linewidth=0)
               for label, color, _ in all_stats]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=11.25, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    save_figure(fig, out_stem)
    plt.close()


def main():
    print("Loading data...")
    # Order: Human first, then models in specified order
    frames = [("Human", HUMAN_COLOR, load_human())]
    for label, path, color in MODELS_ORDERED[1:]:  # Skip the Human placeholder
        print(f"Processing {label}...")
        frames.append((label, color, load_llm(path)))
    all_stats = [(label, color, compute_stats(df)) for label, color, df in frames]
    
    print("Generating figures for Nature Communications with consistent font sizes...")
    print("Font sizes increased by 25% to match other plots")
    print("Model order: Human, GPT-5.2, GPT-5-mini, Gemini-3-flash, Claude Opus 4.6, Qwen3-32B, Llama-3.3-70B-Instruct")
    
    draw_chart_with_errorbars(all_stats,
                              "figures/plots/absolute_belief_change_combined")
    draw_line_with_bands(all_stats,
                         "figures/plots/absolute_belief_change_line_bands")
    draw_dot_range(all_stats,
                   "figures/plots/absolute_belief_change_dot_range")
    dot_range_points([(label, df) for label, _, df in frames]).to_csv(
        "figures/plots/absolute_belief_change_dot_range_nol_points.csv", index=False)

    pct_stats = [("Human", HUMAN_COLOR, compute_percentile_stats(load_human()))]
    for label, path, color in MODELS_ORDERED[1:]:
        pct_stats.append((label, color, compute_percentile_stats(load_llm(path))))
    draw_dot_range_percentile(pct_stats,
                              "figures/plots/absolute_belief_change_dot_range_pct")
    print("Done! All figures saved to figures/plots/")

if __name__ == "__main__":
    main()