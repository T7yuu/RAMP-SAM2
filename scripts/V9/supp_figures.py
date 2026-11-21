"""Supplementary figures for RAMP-SAM2.

Fig. S1  cohort composition: lesion-area fraction and image resolution.
Fig. S2  per-case metric distributions and their dependence on lesion size.

All inputs are measured artefacts: scripts/V9/supp_stats/cohort_raw.npz
(computed from the local ISIC2018 / PH2 directories) and the per-image
confusion matrices under scripts/V9/micro_macro_report/.
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.45,
    }
)

OUT = Path("figures/supplement")
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "ISIC2018-train": "#7f7f7f",
    "ISIC2018-validate": "#2e7d32",
    "ISIC2018-test": "#a50f15",
    "PH2": "#1f5fa8",
}
LBL = {
    "ISIC2018-train": "ISIC2018 train (2594)",
    "ISIC2018-validate": "ISIC2018 val (100)",
    "ISIC2018-test": "ISIC2018 test (1000)",
    "PH2": "PH2 (200)",
}
ORDER = ["ISIC2018-train", "ISIC2018-validate", "ISIC2018-test", "PH2"]


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def load_raw():
    z = np.load("scripts/V9/supp_stats/cohort_raw.npz")
    return z


def fig_s1(z):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))

    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    for k in ORDER:
        f = z[f"{k}_frac"]
        w = np.ones_like(f) / len(f)
        ax.hist(f, bins=bins, weights=w, histtype="step", lw=1.1, color=COL[k], label=LBL[k])
    ax.set_xlabel("Lesion area fraction of the image canvas")
    ax.set_ylabel("Relative frequency")
    ax.set_title("(a) Lesion size distribution", loc="left")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="upper right")
    tidy(ax)

    ax = axes[1]
    bins = np.logspace(np.log10(400), np.log10(8000), 33)
    for k in ORDER:
        s = np.minimum(z[f"{k}_wh"][:, 0], z[f"{k}_wh"][:, 1]).astype(float)
        w = np.ones_like(s) / len(s)
        ax.hist(s, bins=bins, weights=w, histtype="step", lw=1.1, color=COL[k], label=LBL[k])
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([500, 1000, 2000, 4000, 8000]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["500", "1000", "2000", "4000", "8000"]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Short image side (pixels, log scale)")
    ax.set_ylabel("Relative frequency")
    ax.set_title("(b) Native acquisition resolution", loc="left")
    ax.legend(frameon=False, loc="upper right")
    tidy(ax)
    ax.grid(True, axis="x")

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figS1_cohort_statistics.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def load_per_image():
    import csv

    out = {}
    for tag, p in (
        ("ISIC2018-test", "scripts/V9/micro_macro_report/ISIC2018_per_image.csv"),
        ("PH2", "scripts/V9/micro_macro_report/PH2_per_image.csv"),
    ):
        rows = list(csv.DictReader(open(p)))
        out[tag] = {
            "dice": np.array([float(r["dice"]) for r in rows]),
            "iou": np.array([float(r["iou"]) for r in rows]),
            "sens": np.array([float(r["sensitivity"]) for r in rows]),
            "spec": np.array([float(r["specificity"]) for r in rows]),
        }
    return out


def fig_s2(per):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))

    ax = axes[0]
    for k in ("ISIC2018-test", "PH2"):
        d = np.sort(per[k]["dice"])
        ax.plot(d, np.arange(1, len(d) + 1) / len(d), lw=1.2, color=COL[k], label=LBL[k])
    ax.axhline(0.5, color="0.4", lw=0.7, ls=":")
    ax.text(0.02, 0.52, "Dice = 0.5", fontsize=6.3, color="0.35")
    ax.set_xlabel("Case-level Dice")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("(a) Case-level Dice distribution", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    tidy(ax)

    ax = axes[1]
    z = np.load("scripts/V9/supp_stats/cohort_raw.npz")
    frac = z["ISIC2018-test_frac"]
    dice = per["ISIC2018-test"]["dice"]
    n = min(len(frac), len(dice))
    ax.scatter(frac[:n], dice[:n], s=3.2, c=COL["ISIC2018-test"], alpha=0.40, lw=0)
    order = np.argsort(frac[:n])
    nb = 10
    edges = np.array_split(order, nb)
    xs = [np.median(frac[:n][e]) for e in edges]
    ys = [np.median(dice[:n][e]) for e in edges]
    ax.plot(xs, ys, color="black", lw=1.3, marker="o", ms=3.0, label="Binned median")
    ax.set_xlabel("Lesion area fraction")
    ax.set_ylabel("Case-level Dice")
    ax.set_title("(b) Accuracy versus lesion size (ISIC2018 test)", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower left")
    tidy(ax)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figS2_percase_distribution.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    z = load_raw()
    fig_s1(z)
    per = load_per_image()

    # consistency checks against the values reported in the main text
    rep = {}
    for k, v in per.items():
        rep[k] = {
            "n": int(len(v["dice"])),
            "dice_mean": float(v["dice"].mean()),
            "dice_median": float(np.median(v["dice"])),
            "dice_p5": float(np.percentile(v["dice"], 5)),
            "dice_p25": float(np.percentile(v["dice"], 25)),
            "dice_p75": float(np.percentile(v["dice"], 75)),
            "dice_p95": float(np.percentile(v["dice"], 95)),
            "dice_min": float(v["dice"].min()),
            "frac_dice_below_0p5": float((v["dice"] < 0.5).mean()),
            "frac_dice_above_0p9": float((v["dice"] > 0.9).mean()),
        }
    frac = z["ISIC2018-test_frac"]
    dice = per["ISIC2018-test"]["dice"]
    n = min(len(frac), len(dice))
    from scipy.stats import spearmanr

    rho, p = spearmanr(frac[:n], dice[:n])
    rep["ISIC2018-test"]["spearman_dice_vs_area"] = {"rho": float(rho), "p": float(p), "n": int(n)}

    fig_s2(per)
    with open("scripts/V9/supp_stats/figure_numbers.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
