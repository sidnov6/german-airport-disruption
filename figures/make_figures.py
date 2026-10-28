"""Figure panels for the micropublication.

Run from the repository root after the notebooks have written outputs/:

    python figures/make_figures.py

Panels are saved without titles or panel letters (the micropublication guidelines
ask for those to be given in the text).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
DATA = ROOT / "data" / "processed" / "airport_day.parquet"

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
LOGISTIC = "#2a78d6"   # categorical slot 1
LIGHTGBM = "#eb6834"   # slot 2
HISTORY = "#1baf7a"    # slot 3
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#f0efec"
RED = "#e34948"
MASKED = "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": "#8a8984", "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
})

WEATHER = {"Thunderstorm": "thunder_hours", "Low visibility": "low_vis_hours_lt1000",
           "Strong gust": "gust_hours_ge15", "Snow": "snow_hours", "Freezing": "freezing_hours"}
MIN_EVENTS = 10


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load_predictions():
    lr = pd.read_csv(ROOT / "outputs" / "03_logistic_regression" / "test_predictions_2024_2025.csv")
    gb = pd.read_csv(ROOT / "outputs" / "04_lightgbm_classification" / "test_predictions_2024_2025.csv")
    assert (lr["actual"].to_numpy() == gb["actual"].to_numpy()).all()
    return lr, gb


def interpolated_pr(y, score):
    """Precision-recall curve with interpolated precision (best precision at this recall or higher)."""
    p, r, _ = precision_recall_curve(y, score)
    # recall decreases along the arrays (the last point is the artificial recall 0, precision 1)
    return r[:-1], np.maximum.accumulate(p[:-1])


def operating_point(y, score, threshold):
    flag = score >= threshold
    return (flag & (y == 1)).sum() / (y == 1).sum(), (flag & (y == 1)).sum() / flag.sum()


def panel_pr_curves(lr, gb):
    y = lr["actual"].to_numpy()
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    rules = [("LightGBM", gb["probability"].to_numpy(), LIGHTGBM, 3),
             ("Logistic regression", lr["probability"].to_numpy(), LOGISTIC, 2),
             ("Previous 28-day rate", lr["benchmark_28d"].to_numpy(), HISTORY, 1)]
    for name, score, color, z in rules:
        r, p = interpolated_pr(y, score)
        ax.plot(r, p, color=color, linewidth=1.8, zorder=z, solid_capstyle="round",
                label=f"{name}   AP {average_precision_score(y, score):.2f}")
        if name == "LightGBM":
            ax.fill_between(r, p, y.mean(), color=color, alpha=0.08, linewidth=0, zorder=0)
    # alert threshold chosen on 2023
    for name, score, color, _ in rules[:2]:
        rec, prec = operating_point(y, score, 0.20)
        ax.scatter(rec, prec, s=34, color=color, edgecolor="#ffffff", linewidth=1.2, zorder=5)
    rec, prec = operating_point(y, rules[0][1], 0.20)
    ax.annotate("alert threshold 0.20", (rec, prec), xytext=(12, 22), textcoords="offset points",
                fontsize=6.6, color=INK_2, arrowprops=dict(arrowstyle="-", color="#b5b4ae", linewidth=0.6))
    ax.axhline(y.mean(), color="#8a8984", linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)
    ax.text(0.015, y.mean() + 0.018, f"base rate {y.mean():.3f}", ha="left", va="bottom", fontsize=6.6, color=INK_2)
    ax.set_xlabel("Recall: share of disrupted days flagged", fontsize=7.6)
    ax.set_ylabel("Precision: share of flags that were disrupted", fontsize=7.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.2), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)])
    ax.set_yticks(np.arange(0, 1.01, 0.2), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)])
    ax.tick_params(labelsize=7, length=0, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#b5b4ae")
    leg = ax.legend(frameon=False, loc="upper right", fontsize=7.4, handlelength=1.6, borderaxespad=0.2,
                    title="2024-2025 test years", title_fontsize=7.0, alignment="left")
    leg.get_title().set_color(INK_2)
    save(fig, "panelB_precision_recall_2024_2025.png")


def panel_airport_ap(lr, gb):
    rows = []
    for icao, g in lr.groupby("airport_icao"):
        n = int(g["actual"].sum())
        if n < 20:
            continue
        g2 = gb[gb["airport_icao"] == icao]
        rows.append({"airport": f"{g['airport_name'].iloc[0]}\n({n} days)", "n": n,
                     "Logistic regression": average_precision_score(g["actual"], g["probability"]),
                     "LightGBM": average_precision_score(g2["actual"], g2["probability"]),
                     "Previous 28-day rate": average_precision_score(g["actual"], g["benchmark_28d"])})
    t = pd.DataFrame(rows).sort_values("n", ascending=False)
    x = np.arange(len(t))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for k, (col, color) in enumerate([("Logistic regression", LOGISTIC), ("LightGBM", LIGHTGBM),
                                      ("Previous 28-day rate", HISTORY)]):
        ax.bar(x + (k - 1) * w, t[col], width=w - 0.03, color=color, label=col)
    ax.set_xticks(x, t["airport"], fontsize=9)
    ax.set_ylabel("Average precision, 2024-2025")
    ax.set_ylim(0, 0.7)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    save(fig, "extra_average_precision_by_airport.png")


def panel_weather_risk(df):
    ratios = pd.DataFrame(index=sorted(df["airport_name"].unique()), columns=list(WEATHER), dtype=float)
    counts = ratios.copy()
    for airport, g in df.groupby("airport_name"):
        for name, col in WEATHER.items():
            obs = g[g[col].notna()]
            flag = obs[col] > 0
            y = obs["disrupted_day"].astype(float)
            with_w, without_w = y[flag].mean(), y[~flag].mean()
            counts.loc[airport, name] = y[flag].sum()
            if y[flag].sum() >= MIN_EVENTS and without_w > 0:
                ratios.loc[airport, name] = with_w / without_w
    order = ratios.notna().sum(axis=1).sort_values(ascending=False).index
    ratios, counts = ratios.loc[order], counts.loc[order]
    log_ratio = np.log2(ratios.astype(float))
    cmap = LinearSegmentedColormap.from_list("div", [LOGISTIC, NEUTRAL, RED]).with_extremes(bad=MASKED)
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=2.5)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    im = ax.imshow(np.ma.masked_invalid(log_ratio.values), cmap=cmap, norm=norm, aspect="auto")
    for i in range(ratios.shape[0]):
        for j in range(ratios.shape[1]):
            v = ratios.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.1f}×", ha="center", va="center", fontsize=6.2, color=INK)
            else:
                ax.text(j, i, f"n={int(counts.iloc[i, j])}", ha="center", va="center", fontsize=5.2, color=INK_2)
    ax.set_xticks(range(ratios.shape[1]), ["Thunder-\nstorm", "Low\nvisibility", "Strong\ngust", "Snow", "Freezing"], fontsize=6.2)
    ax.set_yticks(range(ratios.shape[0]), ratios.index, fontsize=6.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    ticks = [0.5, 1, 2, 4]
    cb.set_ticks(np.log2(ticks), labels=[f"{t:g}×" for t in ticks])
    cb.set_label("Risk ratio (with / without condition)", fontsize=6.2)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    save(fig, "panelC_weather_risk_ratio_by_airport.png")


def panel_traffic_weather(df):
    d = df.copy()
    observed = d[list(WEATHER.values())].notna().all(axis=1)
    d = d[observed & d["disrupted_day"].notna()]
    median = d.groupby(["airport_icao", "year"])["arrivals"].transform("median")
    d["traffic"] = pd.cut(d["arrivals"] / median, [-np.inf, 0.8, 1.0, 1.2, np.inf],
                          labels=["<80%", "80-100%", "100-120%", ">120%"])
    hazards = sum((d[c] > 0).astype(int) for c in WEATHER.values())
    d["hazards"] = pd.cut(hazards, [-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"])
    rate = d.groupby(["hazards", "traffic"], observed=True)["disrupted_day"].mean().unstack() * 100
    n = d.groupby(["hazards", "traffic"], observed=True).size().unstack()
    shown = rate.where(n >= 50)
    cmap = LinearSegmentedColormap.from_list("seq", BLUE_RAMP).with_extremes(bad=MASKED)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    im = ax.imshow(np.ma.masked_invalid(shown.values.astype(float)), cmap=cmap, vmin=0, vmax=30, aspect="auto")
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            v = shown.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.1f}%\nn={int(n.iloc[i, j]):,}", ha="center", va="center", fontsize=5.8,
                        color="white" if v > 18 else INK)
            else:
                ax.text(j, i, f"n={int(n.iloc[i, j])}", ha="center", va="center", fontsize=5.4, color=INK_2)
    ax.set_xticks(range(shown.shape[1]), shown.columns, fontsize=6.2)
    ax.set_yticks(range(shown.shape[0]), shown.index, fontsize=6.2)
    ax.set_xlabel("Arrivals relative to the airport's median day in the same year", fontsize=6.4)
    ax.set_ylabel("Adverse weather conditions", fontsize=6.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Disrupted airport-days (%)", fontsize=6.2)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    save(fig, "panelD_disruption_by_traffic_and_weather.png")


def panel_pipeline():
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 45)
    ax.axis("off")
    strip = 4.6  # height of the coloured title strip

    def bullets(lines):  # a line starting with two spaces continues the previous bullet
        return "\n".join(f"     {s.strip()}" if s.startswith("  ") else f"\u2022  {s}" for s in lines)

    def card(x, y, w, h, title, lines, color, tint):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                    facecolor=tint, edgecolor="none", zorder=1))
        ax.add_patch(FancyBboxPatch((x, y + h - strip), w, strip, boxstyle="round,pad=0,rounding_size=1.4",
                                    facecolor=color, edgecolor="none", zorder=2))
        ax.add_patch(plt.Rectangle((x, y + h - strip), w, strip / 2, facecolor=color, edgecolor="none", zorder=2))
        ax.text(x + 1.6, y + h - strip / 2, title, ha="left", va="center", fontsize=6.9, fontweight="bold",
                color="#ffffff", zorder=3)
        ax.text(x + 1.6, y + h - strip - 1.8, bullets(lines), ha="left", va="top",
                fontsize=5.4, color=INK, linespacing=1.5, zorder=3)

    def arrow(x1, x2, y):
        ax.annotate("", (x2, y), (x1, y), arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22",
                                                         color="#8a8984", linewidth=1.0, shrinkA=0, shrinkB=0))

    for x, label in [(0, "1  SOURCES"), (26, "2  PROCESSING"), (53, "3  DATASET"), (79, "4  ANALYSIS")]:
        ax.text(x, 43.6, label, fontsize=6.4, fontweight="bold", color="#8a8984", va="center")

    blue, blue_t = LOGISTIC, "#eaf2fc"
    grey, grey_t = "#6f6e69", "#f3f2ee"
    orange, orange_t = LIGHTGBM, "#fdeee6"
    green, green_t = HISTORY, "#e6f6f0"
    card(0, 22, 21.5, 18.5, "EUROCONTROL", ["apt_dly_<year>.csv.bz2", "12 yearly files", "arrivals, ATFM delay", "16 delay causes"], blue, blue_t)
    card(0, 0.5, 21.5, 18.5, "DWD open data", ["hourly station records", "6 products \u00d7 12 stations", "temperature, wind, gusts,", "  rain, visibility, codes"], blue, blue_t)
    card(26, 22, 22.5, 18.5, "Clean traffic", ["keep the 12 airports", "blank delay cell = 0", "causes sum to total", "38 missing days kept"], grey, grey_t)
    card(26, 0.5, 22.5, 18.5, "Clean weather", ["\u2212999 \u2192 missing, UTC check", "hourly \u2192 daily (\u2265 20 h)", "WMO weather codes:", "  fog, thunder, snow, freezing"], grey, grey_t)

    # dataset card with the headline numbers
    x, y, w, h = 53, 0.5, 21.5, 40
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor=orange_t, edgecolor="none", zorder=1))
    ax.add_patch(FancyBboxPatch((x, y + h - strip), w, strip, boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor=orange, edgecolor="none", zorder=2))
    ax.add_patch(plt.Rectangle((x, y + h - strip), w, strip / 2, facecolor=orange, edgecolor="none", zorder=2))
    ax.text(x + 1.6, y + h - strip / 2, "airport_day table", ha="left", va="center", fontsize=6.9,
            fontweight="bold", color="#ffffff", zorder=3)
    ax.text(x + w / 2, 28.6, "52,596", ha="center", va="center", fontsize=13, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, 24.6, "airport-days", ha="center", va="center", fontsize=6.4, color=INK_2, zorder=3)
    ax.text(x + 1.6, 20.6, bullets(["12 airports × 4,383 days", "79 columns, 37 predictors",
            "target: ≥ 2% of arrivals", "  delayed > 15 min", "history: earlier days only"]),
            ha="left", va="top", fontsize=5.4, color=INK, linespacing=1.5, zorder=3)

    card(79, 22, 21, 18.5, "BI notebooks", ["KPIs as ratios of sums", "delay causes by airport", "weather risk ratios", "traffic \u00d7 weather"], green, green_t)
    card(79, 0.5, 21, 18.5, "Models", ["train 2014\u20132021", "tune 2022, calibrate 2023", "test 2024\u20132025, once", "logistic reg. + LightGBM"], green, green_t)

    for yy in (31.25, 9.75):
        arrow(21.9, 25.6, yy)
        arrow(48.9, 52.6, yy)
        arrow(74.9, 78.6, yy)
    save(fig, "panelE_data_pipeline.png")


AIRPORT_LABELS = {  # display name, label offset in points (dx, dy)
    "EDDF": ("Frankfurt", (17, -4)), "EDDM": ("Munich", (15, -11)), "EDDL": ("Düsseldorf", (-8, 4)),
    "EDDH": ("Hamburg", (9, 3)), "EDDK": ("Cologne-Bonn", (-8, -9)), "EDDS": ("Stuttgart", (-8, -7)),
    "EDDP": ("Leipzig-Halle", (8, 3)), "EDDV": ("Hannover", (7, -7)), "EDDN": ("Nuremberg", (7, 3)),
    "EDDW": ("Bremen", (-7, 4)), "EDDC": ("Dresden", (7, -6)), "EDDG": ("Münster-Osnabrück", (-7, 6)),
}


def panel_map(df):
    import json
    import matplotlib.patheffects as pe
    from matplotlib.patches import Polygon
    import yaml

    geo = json.loads((ROOT / "data" / "geo" / "germany_states_neighbours.geojson").read_text())
    airports = pd.DataFrame(yaml.safe_load((ROOT / "pipeline" / "config.yaml").read_text())["airports"])
    stats = df.groupby("airport_icao").agg(arrivals=("arrivals", "sum"), rate=("disrupted_day", "mean"))
    airports = airports.join(stats, on="icao")

    fig, ax = plt.subplots(figsize=(3.6, 4.4))
    ax.set_facecolor("#e4edf6")  # sea
    style = {  # layer: facecolor, edgecolor, linewidth, zorder
        "neighbour": ("#f7f6f3", "#d9d8d2", 0.5, 1),
        "state": ("#e9e1d3", "#fbf8f3", 0.6, 2),  # Germany tinted, light lines between states
        "germany": ("none", "#5f5e5a", 0.9, 3),   # national border on top
    }
    for f in geo["features"]:
        face, edge, lw, z = style[f["properties"]["layer"]]
        for poly in f["geometry"]["coordinates"]:
            ax.add_patch(Polygon(poly[0], closed=True, facecolor=face, edgecolor=edge, linewidth=lw, zorder=z))
    ax.set_xlim(5.5, 15.4)
    ax.set_ylim(47.15, 55.15)
    ax.set_aspect(1 / np.cos(np.radians(51.2)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ramp = LinearSegmentedColormap.from_list("disruption", ["#fbe3d6", "#f4a07a", "#eb6834", "#b8461a", "#7a2a0b"])
    size = lambda a: 640 * a / 2.6e6  # noqa: E731  marker area proportional to arrivals
    sc = ax.scatter(airports["lon"], airports["lat"], s=size(airports["arrivals"]), c=airports["rate"] * 100,
                    cmap=ramp, vmin=0, vmax=20, edgecolors="#ffffff", linewidths=0.8, zorder=4)
    halo = [pe.withStroke(linewidth=2.2, foreground="#ffffff")]
    for _, a in airports.iterrows():
        name, (dx, dy) = AIRPORT_LABELS[a["icao"]]
        ha = "left" if dx >= 0 else "right"
        ax.annotate(f"{name}\n{a['rate'] * 100:.1f}%", (a["lon"], a["lat"]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=6.6, color=INK,
                    linespacing=1.05, path_effects=halo, zorder=5)

    # white cards behind the two legends so borders do not run through them
    for x, y, w, h in [(0.70, 0.80, 0.285, 0.19), (0.025, 0.012, 0.44, 0.115)]:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
                                    transform=ax.transAxes, facecolor="#ffffff", edgecolor="#d9d8d2",
                                    linewidth=0.4, alpha=0.92, zorder=4))
    # nested size legend: circles share a bottom edge
    fig.canvas.draw()
    pts_per_deg = (ax.transData.transform((0, 1)) - ax.transData.transform((0, 0)))[1] * 72 / fig.dpi
    x0, base = 13.72, 53.9
    for val, lab in [(2.5e6, "2.5 M"), (1.0e6, "1 M"), (0.25e6, "0.25 M")]:
        r = np.sqrt(size(val)) / 2 / pts_per_deg  # scatter s is diameter squared in points
        ax.scatter(x0, base + r, s=size(val), facecolor="none", edgecolor=INK_2, linewidth=0.7, zorder=5)
        ax.plot([x0, x0 + 0.5], [base + 2 * r] * 2, color=INK_2, linewidth=0.5, zorder=5)
        ax.text(x0 + 0.55, base + 2 * r, lab, va="center", fontsize=5.6, color=INK_2, zorder=5)
    ax.text(x0 + 0.2, 54.95, "Arrivals 2014-2025", ha="center", fontsize=5.8, color=INK_2, zorder=5)
    cax = ax.inset_axes([0.06, 0.05, 0.36, 0.022])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 5, 10, 15, 20], labels=["0", "5", "10", "15", "20%"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.8, length=2, colors=INK_2)
    cax.set_title("Disrupted airport-days, 2014-2025", fontsize=5.8, color=INK_2, loc="left", pad=2)
    ax.text(0.99, 0.005, "Boundaries: Natural Earth", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=4.6, color="#8a8984", zorder=5)
    save(fig, "panelA_airport_map.png")


def main():
    lr, gb = load_predictions()
    df = pd.read_parquet(DATA)
    df["disrupted_day"] = df["disrupted_day"].astype("float64")
    for c in ["arrivals", *WEATHER.values()]:
        df[c] = df[c].astype("float64")
    panel_map(df)
    panel_pr_curves(lr, gb)
    panel_airport_ap(lr, gb)
    panel_weather_risk(df)
    panel_traffic_weather(df)
    panel_pipeline()


if __name__ == "__main__":
    main()
