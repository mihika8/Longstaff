"""
setting up (scenario, frac) combinations and pricing an American
put and compare against the polynomial LSM baseline. 
testing 3 regimes that will stress test the LOWESS estimator 
in different ways:
1. Baseline GBM 
2. High jump intensity 
3. Deep OTM (out-of-the-money pathways)


frac sets up how many of the nearest points the local fit uses
higher frac values --> smoother and more global fit 
"""

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings, time
warnings.filterwarnings("ignore")

N_SEEDS = 8 
SEEDS = list(range(42, 42 + N_SEEDS))

BASE = dict(r=0.02, sigma=0.20, T=1.0, N=50, K=100)

SCENARIOS = {
    "Baseline GBM\n(S=100, λ=0)":       dict(S0=100, lam=0.0, m_j=0.0,  v_j=0.0),
    "High Jump Intensity\n(S=100, λ=1)": dict(S0=100, lam=1.0, m_j=-0.02, v_j=0.04),
    "Deep OTM\n(S=120, λ=0)":            dict(S0=120, lam=0.0, m_j=0.0,  v_j=0.0),
}

M = 5_000
FRAC_GRID = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70]
POLY_DEG  = 2

#simulation
def simulate_paths(S0, r, sigma, T, N, M, lam, m_j, v_j, seed):
    np.random.seed(seed)
    dt = T/N
    paths = np.zeros((N+1, M))
    paths[0] = S0 

    Z = np.random.normal(size=(N,M))
    K_arr = np.random.poisson(lam*dt, size=(N,M)) if lam >0 else np.zeros((N,M), int)

    kappa = np.exp(m_j+0.5*v_j)-1.0 if lam>0 else 0.0
    max_K = int(K_arr.max())

    Y = None 
    if max_K > 0:
        Y = np.random.normal(loc=m_j, scale=np.sqrt(v_j), size=(max_K, N,M))

    drift = (r-lam*kappa-0.5*sigma**2)*dt 
    vol = sigma*np.sqrt(dt)

    for t in range(1, N+1):
        S_prev = paths[t-1]
        S_new = S_prev*np.exp(drift+vol*Z[t-1])

        if max_K > 0:
            K_t = K_arr[t-1]
            jump_sum = np.zeros(M)
            for k in range(1, max_K+1):
                mask= K_t>=k
                if mask.any():
                    jump_sum[mask]+=Y[k-1, t-1, mask]
            S_new = S_new * np.exp(jump_sum)

        paths[t] = S_new
    return paths 

def lsm_price(paths, r, T, K, method="poly", poly_degree=2,
              lowess_frac=0.25, lowess_it=0):
    N_steps = paths.shape[0]-1
    dt = T/N_steps
    #setting up discount factor 
    disc = np.exp(-r*dt)

    #setting up cashflows starting with the terminal payoff 
    #backward induction: at each earlier timstep check - should I exercise now or keep holding? 
    cf = np.maximum(K-paths[-1], 0.0)

    for t in range(N_steps-1, 0, -1):
        S_t = paths[t] #defining stock price at time t based on a given simulated path 
        immediate = np.maximum(K-S_t, 0.0) #calculating value of the call (intrinsic value formual)
        itm = immediate > 0 
        if not itm.any():
            cf*=disc
            continue 
        #discounted future cashflow from continuing to hold 
        cont_disc = cf *disc 
        x = S_t[itm]; y = cont_disc[itm]

        if method == "lowess":
            if len(x) >= 20:
                order = np.argsort(x)
                yhat_s = lowess(endog=y[order], exog=x[order],
                                frac=lowess_frac, it=lowess_it,
                                return_sorted=False)
                cont_itm = np.empty_like(yhat_s); cont_itm[order] = yhat_s
            else:
                X = np.column_stack([np.ones_like(x), x])
                cont_itm = X @ np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            X = np.column_stack([x**d for d in range(poly_degree + 1)])
            cont_itm = X @ np.linalg.lstsq(X, y, rcond=None)[0]

        cont_all = np.zeros_like(S_t)
        cont_all[itm] = cont_itm

        #exercise decision 
        #for each path if in the money AND immediate payoff beats continuation value --> exercise 
        #else keep the discounted future cashflow (updating cf for the next iteration)
        ex = itm & (immediate > cont_all)
        cf[ex]  = immediate[ex]
        cf[~ex] = cont_disc[~ex]

    #last discount after using backwards induction to get back to t=1
    #average these values across the simulated stock pthats to get the price estimate
    return disc * cf.mean()

print("Running frac sweep stress test …")
print(f"  {len(SCENARIOS)} scenarios × {len(FRAC_GRID)} frac values × {N_SEEDS} seeds\n")
 
results = {}   # scenario → {frac: [prices], "poly": [prices]}
 
for sc_name, sc_params in SCENARIOS.items():
    print(f"  Scenario: {sc_name.replace(chr(10),' ')}")
    r_prices = {"poly": []}
    for frac in FRAC_GRID:
        r_prices[frac] = []
 
    for seed in SEEDS:
        paths = simulate_paths(
            S0=sc_params["S0"], r=BASE["r"], sigma=BASE["sigma"],
            T=BASE["T"], N=BASE["N"], M=M,
            lam=sc_params["lam"], m_j=sc_params["m_j"], v_j=sc_params["v_j"],
            seed=seed
        )
        # polynomial baseline
        r_prices["poly"].append(
            lsm_price(paths, BASE["r"], BASE["T"], BASE["K"], method="poly")
        )
        # LOWESS at each frac
        for frac in FRAC_GRID:
            r_prices[frac].append(
                lsm_price(paths, BASE["r"], BASE["T"], BASE["K"],
                          method="lowess", lowess_frac=frac)
            )
 
    results[sc_name] = r_prices
    poly_mean = np.mean(r_prices["poly"])
    print(f"    Poly baseline: {poly_mean:.4f}")
    for frac in FRAC_GRID:
        lo_mean = np.mean(r_prices[frac])
        print(f"    frac={frac:.2f}: {lo_mean:.4f}  (Δ={lo_mean-poly_mean:+.4f})")
    print()
 
print("Done. Building plots …")
 
 
# ── plotting ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 13))
fig.patch.set_facecolor("#f8f9fa")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
 
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(FRAC_GRID)))
sc_list = list(SCENARIOS.keys())
 
# ── top row: mean price vs frac ──────────────────────────────────────────────
for col, sc_name in enumerate(sc_list):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor("white")
    r = results[sc_name]
 
    poly_mean = np.mean(r["poly"])
    poly_std  = np.std(r["poly"])
 
    means = [np.mean(r[f]) for f in FRAC_GRID]
    stds  = [np.std(r[f])  for f in FRAC_GRID]
 
    ax.axhline(poly_mean, color="#e74c3c", lw=2.0, ls="--", label=f"Poly (deg {POLY_DEG})")
    ax.fill_between(range(len(FRAC_GRID)),
                    poly_mean - poly_std, poly_mean + poly_std,
                    color="#e74c3c", alpha=0.12)
 
    ax.plot(range(len(FRAC_GRID)), means, "o-", color="#2c7bb6", lw=2, ms=6, label="LOWESS mean")
    ax.fill_between(range(len(FRAC_GRID)),
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    color="#2c7bb6", alpha=0.18, label="±1 SD")
 
    ax.set_xticks(range(len(FRAC_GRID)))
    ax.set_xticklabels([str(f) for f in FRAC_GRID], fontsize=8, rotation=45)
    ax.set_xlabel("LOWESS frac", fontsize=10)
    ax.set_ylabel("Put Price", fontsize=10)
    ax.set_title(sc_name, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, framealpha=0.7)
    ax.grid(True, alpha=0.3, lw=0.5)
 
# ── bottom row: price deviation from poly ────────────────────────────────────
for col, sc_name in enumerate(sc_list):
    ax = fig.add_subplot(gs[1, col])
    ax.set_facecolor("white")
    r = results[sc_name]
 
    poly_mean = np.mean(r["poly"])
    means = np.array([np.mean(r[f]) for f in FRAC_GRID])
    stds  = np.array([np.std(r[f])  for f in FRAC_GRID])
    devs  = means - poly_mean
 
    bar_colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in devs]
    bars = ax.bar(range(len(FRAC_GRID)), devs, color=bar_colors, alpha=0.75, edgecolor="white", lw=0.8)
 
    ax.errorbar(range(len(FRAC_GRID)), devs, yerr=stds,
                fmt="none", color="#2c3e50", capsize=4, lw=1.2, alpha=0.6)
 
    ax.axhline(0, color="black", lw=1.0, ls="-")
    ax.set_xticks(range(len(FRAC_GRID)))
    ax.set_xticklabels([str(f) for f in FRAC_GRID], fontsize=8, rotation=45)
    ax.set_xlabel("LOWESS frac", fontsize=10)
    ax.set_ylabel("LOWESS − Poly", fontsize=10)
    ax.set_title(f"Deviation from Poly Baseline\n{sc_name}", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3, lw=0.5, axis="y")
 
    for bar, dev in zip(bars, devs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                dev + (0.001 if dev >= 0 else -0.002),
                f"{dev:+.3f}", ha="center", va="bottom" if dev >= 0 else "top",
                fontsize=6.5, color="#2c3e50")
 
fig.suptitle(
    "LOWESS Bandwidth (frac) Stress Test vs Polynomial LSM Baseline\n"
    f"M={M:,} paths, {N_SEEDS} seeds averaged, T=1yr, K=100",
    fontsize=13, fontweight="bold", y=1.01
)
 
plt.savefig("frac_sweep_stress_test.png",
            dpi=150, bbox_inches="tight")
print("Saved → frac_sweep_stress_test.png")