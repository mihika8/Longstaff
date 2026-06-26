#!/usr/bin/env python3
"""
Create summary tables and figures from LSM experiment CSV outputs.

Expected input files in ./output:
- lsm_full_results_fast.csv
- lsm_summary_fast.csv
- lsm_boundary_summary_fast.csv (optional)
- lsm_lowess_frac_summary_fast.csv (optional)

Outputs:
- results_table.csv
- price_by_scenario.png
- eep_by_scenario.png
- boundary_plot.png (if boundary file exists)
- lowess_frac_plot.png (if frac file exists)
- results_report.txt

Run:
    python lsm_results_plots.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

print("[START] Building result plots...", flush=True)

OUTPUT_DIR = "output"
FULL_RESULTS = os.path.join(OUTPUT_DIR, "lsm_full_results_fast.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "lsm_summary_fast.csv")
BOUNDARY_FILE = os.path.join(OUTPUT_DIR, "lsm_boundary_summary_fast.csv")
FRAC_FILE = os.path.join(OUTPUT_DIR, "lsm_lowess_frac_summary_fast.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def require_file(path):
    if not os.path.exists(path):
        print(f"[ERROR] Missing required file: {path}", flush=True)
        sys.exit(1)


def scenario_order_key(name):
    order = {"A1": 0, "A2": 1, "B2_fixed": 2, "B2_cv": 3}
    return order.get(name, 999)


require_file(FULL_RESULTS)
require_file(SUMMARY_FILE)

print("[LOAD] Reading CSV files...", flush=True)
full = pd.read_csv(FULL_RESULTS)
summary = pd.read_csv(SUMMARY_FILE)
boundary = pd.read_csv(BOUNDARY_FILE) if os.path.exists(BOUNDARY_FILE) else None
frac = pd.read_csv(FRAC_FILE) if os.path.exists(FRAC_FILE) else None

summary = summary.sort_values("scenario", key=lambda s: s.map(scenario_order_key))

print("[TABLE] Creating compact summary table...", flush=True)
results_table = summary[
    [
        "scenario", "model", "lambda", "regression", "span_method",
        "american_mean", "american_sd_across_reps", "avg_mc_se",
        "european_mean", "eep_mean", "ci95_halfwidth"
    ]
].copy()

for col in ["american_mean", "american_sd_across_reps", "avg_mc_se", "european_mean", "eep_mean", "ci95_halfwidth"]:
    results_table[col] = results_table[col].round(6)

results_table.to_csv(os.path.join(OUTPUT_DIR, "results_table.csv"), index=False)

plt.style.use("seaborn-v0_8-whitegrid")

print("[PLOT] American price by scenario...", flush=True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(results_table["scenario"], results_table["american_mean"], color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][:len(results_table)])
ax.errorbar(results_table["scenario"], results_table["american_mean"],
            yerr=results_table["ci95_halfwidth"], fmt='none', ecolor='black', capsize=5)
ax.set_title("American put price by scenario")
ax.set_xlabel("Scenario")
ax.set_ylabel("Mean American price")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "price_by_scenario.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("[PLOT] Early exercise premium by scenario...", flush=True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(results_table["scenario"], results_table["eep_mean"], color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][:len(results_table)])
ax.set_title("Early exercise premium by scenario")
ax.set_xlabel("Scenario")
ax.set_ylabel("Mean early exercise premium")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eep_by_scenario.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

if boundary is not None and not boundary.empty:
    print("[PLOT] Exercise boundary figure...", flush=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for scen in sorted(boundary["scenario"].unique(), key=scenario_order_key):
        sub = boundary[boundary["scenario"] == scen].sort_values("time")
        ax.plot(sub["time"], sub["boundary_mean"], label=scen, linewidth=2)
    ax.set_title("Estimated exercise boundary over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Boundary stock price")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boundary_plot.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
else:
    print("[SKIP] No boundary summary file found.", flush=True)

if frac is not None and not frac.empty:
    print("[PLOT] LOWESS chosen span figure...", flush=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for scen in sorted(frac["scenario"].unique(), key=scenario_order_key):
        sub = frac[frac["scenario"] == scen].sort_values("time")
        ax.plot(sub["time"], sub["frac_mean"], label=scen, linewidth=2)
    ax.set_title("Chosen LOWESS span over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean chosen LOWESS frac")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lowess_frac_plot.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
else:
    print("[SKIP] No LOWESS fraction summary file found.", flush=True)

print("[REPORT] Writing text summary...", flush=True)
with open(os.path.join(OUTPUT_DIR, "results_report.txt"), "w", encoding="utf-8") as f:
    f.write("LSM experiment results summary\n")
    f.write("=============================\n\n")
    f.write("Scenario summary table:\n\n")
    f.write(results_table.to_string(index=False))
    f.write("\n\nQuick interpretation notes:\n")
    if len(results_table) >= 2:
        f.write("- Compare A1 vs A2 for the effect of adding jumps.\n")
    if "B2_fixed" in results_table["scenario"].values:
        f.write("- Compare A2 vs B2_fixed for polynomial vs fixed-LOWESS under jumps.\n")
    if "B2_cv" in results_table["scenario"].values:
        f.write("- Compare B2_fixed vs B2_cv for fixed span vs CV-selected LOWESS.\n")

print("[DONE] Plots and tables written to output/", flush=True)