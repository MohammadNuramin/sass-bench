#!/usr/bin/env python3
"""
Publication figures for "Valid JSON, False Facts".
Applies the dataviz method: form by job, validated CVD-safe palette (blue<->red
diverging for polarity, single-hue blue sequential for magnitude), recessive chrome,
direct labels. Outputs 300-DPI PNG + vector PDF to results_placeholder/figures/.
"""
import csv, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ---- validated palette (dataviz skill reference instance) ----
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, RED, AQUA = "#2a78d6", "#d03b3b", "#1baf7a"
SEQ = ["#eef5fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12.5,
    "axes.titleweight": "bold", "axes.labelsize": 10.5, "axes.labelcolor": INK2,
    "text.color": INK, "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "figure.dpi": 120, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results_placeholder")
FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)

SHORT = {
    "LiquidAI/LFM2.5-230M": "LFM2.5-230M", "LiquidAI/LFM2.5-350M": "LFM2.5-350M",
    "Qwen/Qwen3.5-0.8B": "Qwen3.5-0.8B", "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
    "tiiuae/Falcon-H1-Tiny-Multilingual-100M-Instruct": "Falcon-H1-100M",
    "tiiuae/Falcon-H1-Tiny-R-0.6B": "Falcon-H1-R-0.6B",
    "ibm-granite/granite-4.0-350m": "Granite-4.0-350M",
    "ibm-granite/granite-4.0-h-350m": "Granite-4.0-H-350M",
    "google/functiongemma-270m-it": "FunctionGemma-270M",
    "LiquidAI/LFM2-350M-Extract": "LFM2-Extract-350M",
}
def rd(p):
    return list(csv.DictReader(open(p)))
def num(x):
    try: return float(x)
    except: return np.nan
def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300)
    plt.close(fig); print("  wrote", name + ".png/.pdf")
def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)

# fair-baseline scoring (C0 JSON-aware) so LFM2-Extract's baseline is correct
CI = {r["model"]: r for r in rd(os.path.join(RES, "c0robust", "spi_confidence_intervals.csv"))}
SUM = {(r["model"], r["condition"]): r for r in rd(os.path.join(RES, "c0robust", "summary_by_model_condition.csv"))}
base = [m for m in CI if "@" not in m]

# ======================================================================= FIG 1
# Job: magnitude + polarity + uncertainty -> horizontal forest plot, diverging color.
rows = sorted(base, key=lambda m: num(CI[m]["spi_c2"]))
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for i, m in enumerate(rows):
    r = CI[m]; spi = num(r["spi_c2"]); lo = num(r["spi_c2_ci_lo"]); hi = num(r["spi_c2_ci_hi"])
    sig = str(r["spi_c2_significant_positive"]).strip().lower() == "true"
    col = RED if sig else MUTED
    ax.plot([lo, hi], [i, i], color=col, lw=2.4, solid_capstyle="round", zorder=2)
    ax.scatter([spi], [i], s=54, color=col, zorder=3, edgecolor=SURF, linewidth=1.2)
    ax.annotate(f"{spi:+.2f}", (hi, i), xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9, color=INK2)
ax.axvline(0, color=BASE, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([SHORT[m] for m in rows], fontsize=10)
ax.set_xlabel("Schema Pressure Index   (SPI-C2 = FalseFill$_{strict\\,JSON}$ − FalseFill$_{free\\,text}$)")
ax.set_title("Strict-JSON prompting raises fabrication in capable sub-1B SLMs", pad=10)
ax.margins(x=0.14); despine(ax)
ax.set_axisbelow(True); ax.grid(axis="x", color=GRID, lw=0.7)
ax.legend(handles=[Line2D([0],[0],marker="o",color=RED,lw=2.4,ms=7,label="schema pressure (95% CI excludes 0)"),
                   Line2D([0],[0],marker="o",color=MUTED,lw=2.4,ms=7,label="not significant")],
          loc="lower right", frameon=False, fontsize=8.6, handlelength=1.6)
ax.text(0, len(rows)-0.4, "  more fabrication under schema →", color=MUTED, fontsize=8.2, ha="left", va="center")
save(fig, "fig1_schema_pressure_index")

# ======================================================================= FIG 2
# Job: magnitude across an ordered grid (model x condition) -> sequential heatmap.
conds = ["C0", "C2", "C3", "C4", "C5"]
clabel = ["C0\nfree-text", "C2\nstrict", "C3\n+null", "C4\nfew-shot", "C5\nconstrained"]
order = sorted(base, key=lambda m: -num(SUM.get((m, "C2"), {}).get("false_fill_rate", "nan")))
M = np.array([[num(SUM.get((m, c), {}).get("false_fill_rate", "nan")) for c in conds] for m in order])
fig, ax = plt.subplots(figsize=(6.6, 5.0))
im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(conds))); ax.set_xticklabels(clabel, fontsize=9.2)
ax.set_yticks(range(len(order))); ax.set_yticklabels([SHORT[m] for m in order], fontsize=9.6)
for i in range(len(order)):
    for j in range(len(conds)):
        v = M[i, j]
        if np.isnan(v):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color="#f0efec")); txt = "–"; c = MUTED
        else:
            txt = f"{v:.2f}"; c = "white" if v > 0.55 else INK
        ax.text(j, i, txt, ha="center", va="center", fontsize=8.6, color=c)
ax.set_xticks(np.arange(-.5, len(conds), 1), minor=True); ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
ax.grid(which="minor", color=SURF, lw=2.2); ax.tick_params(which="minor", length=0)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("False Fill Rate", fontsize=9.5)
cb.outline.set_visible(False); cb.ax.tick_params(color=MUTED, labelsize=8.5)
ax.set_title("False-fill rate across prompting conditions", pad=10)
save(fig, "fig2_ffr_by_condition_heatmap")

# ======================================================================= FIG 3
# Job: two measures per entity under C5 -> grouped bars (format vs truth decoupling).
c5m = [m for m in base if not np.isnan(num(SUM.get((m, "C5"), {}).get("schema_validity_rate", "nan")))]
c5m = sorted(c5m, key=lambda m: num(SUM[(m, "C5")]["false_fill_rate"]))
sv = [num(SUM[(m, "C5")]["schema_validity_rate"]) for m in c5m]
ff = [num(SUM[(m, "C5")]["false_fill_rate"]) for m in c5m]
y = np.arange(len(c5m)); h = 0.38
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.barh(y+h/2, sv, height=h, color=BLUE, label="schema validity (structure)", zorder=3)
ax.barh(y-h/2, ff, height=h, color=RED, label="false-fill rate (fabrication)", zorder=3)
for i, (a, b) in enumerate(zip(sv, ff)):
    ax.text(a+0.012, i+h/2, f"{a:.2f}", va="center", fontsize=8.3, color=INK2)
    ax.text(b+0.012, i-h/2, f"{b:.2f}", va="center", fontsize=8.3, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([SHORT[m] for m in c5m], fontsize=9.6)
ax.set_xlim(0, 1.10); ax.set_xlabel("rate")
ax.set_title("Constrained decoding (C5) guarantees valid JSON — not truth", pad=26)
despine(ax); ax.set_axisbelow(True); ax.grid(axis="x", color=GRID, lw=0.7)
# legend above the plot (bars span full width -> no in-plot space)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False,
          fontsize=9.5, handlelength=1.4, columnspacing=1.8)
save(fig, "fig3_c5_format_truth_gap")

# ======================================================================= FIG 4
# Job: agreement between two measurements -> scatter with y=x reference.
import statistics as st
GN = {"MISSING", "AMBIGUOUS", "CONTRADICTORY"}
def ffr_reps(fm_csv, cond):
    agg = {}
    for r in rd(fm_csv):
        if r["condition"] != cond or r["field_status"] not in GN: continue
        b = r["model"].split("@")[0]; rep = r["model"].split("@robust-r")[-1]
        a = agg.setdefault((b, rep), [0, 0]); a[0] += int(r["is_false_fill"] or 0); a[1] += 1
    out = {}
    for (b, rep), (f, t) in agg.items():
        out.setdefault(b, {})[rep] = f/t if t else np.nan
    return out
c0 = ffr_reps(os.path.join(RES, "robustness", "field_metrics.csv"), "C0")
c2 = ffr_reps(os.path.join(RES, "robust_c2", "field_metrics.csv"), "C2")
sigm = ["Qwen/Qwen3-0.6B","Qwen/Qwen3.5-0.8B","ibm-granite/granite-4.0-350m","ibm-granite/granite-4.0-h-350m","LiquidAI/LFM2.5-350M"]
fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.plot([-0.05, 0.6], [-0.05, 0.6], color=BASE, lw=1.2, ls=(0, (4, 3)), zorder=1)
for m in sigm:
    if m not in c0 or m not in c2: continue
    det = num(CI[m]["spi_c2"]); reps = sorted(set(c0[m]) & set(c2[m]))
    spis = [c2[m][r]-c0[m][r] for r in reps]; mean = st.mean(spis); sd = st.pstdev(spis) if len(spis) > 1 else 0
    ax.errorbar(det, mean, yerr=max(sd, 0.004), fmt="o", ms=8, color=BLUE, ecolor=INK2,
                elinewidth=1.1, capsize=3, zorder=3, mec=SURF, mew=1.2)
    ax.annotate(SHORT[m], (det, mean), xytext=(7, -3), textcoords="offset points", fontsize=8.3, color=INK2)
ax.set_xlabel("deterministic SPI-C2 (greedy, temp 0)")
ax.set_ylabel("sampled SPI-C2 (temp 0.2, mean of 3 reps)")
ax.set_title("The effect replicates under stochastic sampling", pad=10)
ax.text(0.02, 0.55, "y = x", color=MUTED, fontsize=9, rotation=38)
ax.set_xlim(-0.02, 0.58); ax.set_ylim(-0.02, 0.58); ax.set_aspect("equal")
despine(ax); ax.set_axisbelow(True); ax.grid(color=GRID, lw=0.7)
save(fig, "fig4_replication_scatter")

print("Figures written to", FIG)
