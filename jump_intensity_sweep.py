'''
testing different lambda values with computing prices under both methods 
recording distributional statistics of terminal stock price 
correlate pricing divergence with distributional shape 


'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import skew, kurtosis
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings("ignore")

#average number of jumps per year 
LAM_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
N_SEEDS = 8 
SEEDS = list(range(42, 42 + N_SEEDS))

BASE = dict(r=0.02, sigma=0.20, T=1.0, N=50, K=100, S0=100)
M         = 5_000
FRAC      = 0.25 
V_J = 0.04 
REGIMES = {"Zero-Mean Jumps (m_j=-v_j/2)": dict(m_j=-V_J/2)}

#simulations
def simulate_paths(S0, r, sigma, T, N, M, lam, m_j, v_j, seed):
    np.random.seed(seed)
    dt    = T / N
    #kappa is the expected relative jump size 
    kappa = np.exp(m_j + 0.5 * v_j) - 1.0 if lam > 0 else 0.0
    paths = np.zeros((N + 1, M))
    paths[0] = S0
 
    Z     = np.random.normal(size=(N, M))
    K_arr = np.random.poisson(lam * dt, size=(N, M)) if lam > 0 \
            else np.zeros((N, M), int)
    max_K = int(K_arr.max())
    Y     = np.random.normal(loc=m_j, scale=np.sqrt(v_j),
                              size=(max_K, N, M)) if max_K > 0 else None
 
    drift = (r - lam * kappa - 0.5 * sigma**2) * dt
    vol   = sigma * np.sqrt(dt)
 
    for t in range(1, N + 1):
        S_new = paths[t - 1] * np.exp(drift + vol * Z[t - 1])
        if max_K > 0:
            K_t = K_arr[t - 1]
            jsum = np.zeros(M)
            for k in range(1, max_K + 1):
                mask = K_t >= k
                if mask.any():
                    jsum[mask] += Y[k - 1, t - 1, mask]
            S_new = S_new * np.exp(jsum)
        paths[t] = S_new
    return paths
 
 #LSM pricing 
def lsm_price(paths, r, T, K, method="poly", frac=0.25):
    N_steps = paths.shape[0]-1
    dt = T/N_steps 
    disc = np.exp(-r*dt)
    cf = np.maximum(K-paths[-1], 0.0)

    for t in range(N_steps -1, 0, -1):
        S_t = paths[t]
        immediate = np.maximum(K-S_t, 0.0)
        #is the path in-the-money at t 
        itm = immediate > 0 
        if not itm.any():
            cf*=disc
            continue 

        cont_disc = cf*disc
        x = S_t[itm]; y = cont_disc[itm]

        if method == "lowess" and len(x) >= 20:
            order = np.argsort(x)
            yhat_s = lowess(endog=y[order], exog=x[order],
                              frac=frac, it=0, return_sorted=False)
            cont_itm = np.empty_like(yhat_s); cont_itm[order] = yhat_s
        else:
            X        = np.column_stack([x**d for d in range(3)])
            cont_itm = X @ np.linalg.lstsq(X, y, rcond=None)[0]
 
        cont_all = np.zeros_like(S_t)
        cont_all[itm] = cont_itm
        ex       = itm & (immediate > cont_all)
        cf[ex]   = immediate[ex]
        cf[~ex]  = cont_disc[~ex]
 
    return disc * cf.mean()
 
#at the end of each simulation, capture the shape of the terminal stock price distribution
#trying to answer: do prices differ because the distribution is more fat-tailed

def terminal_stats(paths):
    ST = paths[-1]
    return dict(
        mean = ST.mean(), 
        std = ST.std(ddof=1), 
        skew = float(skew(ST)),
        exkurt = float(kurtosis(ST, fisher = True)),
    )

print("Running jump intensity sweep …")
print(f"  λ grid: {LAM_GRID}")
print(f"  {len(REGIMES)} regimes × {len(LAM_GRID)} λ values × {N_SEEDS} seeds\n")
 
all_results = {}   # regime → {lam: {"poly": [...], "lowess": [...], "stats": {...}}}
 
for reg_name, reg_params in REGIMES.items():
    print(f"  Regime: {reg_name.replace(chr(10),' ')}")
    reg_res = {}
    m_j = reg_params["m_j"]
 
    for lam in LAM_GRID:
        poly_prices  = []
        low_prices   = []
        stat_list    = []
 
        for seed in SEEDS:
            paths = simulate_paths(
                S0=BASE["S0"], r=BASE["r"], sigma=BASE["sigma"],
                T=BASE["T"], N=BASE["N"], M=M,
                lam=lam, m_j=m_j, v_j=V_J, seed=seed
            )
            poly_prices.append(lsm_price(paths, BASE["r"], BASE["T"], BASE["K"],
                                         method="poly"))
            low_prices.append(lsm_price(paths, BASE["r"], BASE["T"], BASE["K"],
                                        method="lowess", frac=FRAC))
            stat_list.append(terminal_stats(paths))
 
        # average distributional stats across seeds
        avg_stats = {k: np.mean([s[k] for s in stat_list]) for k in stat_list[0]}
        reg_res[lam] = dict(poly=poly_prices, lowess=low_prices, stats=avg_stats)
 
        pm = np.mean(poly_prices); lm = np.mean(low_prices)
        print(f"    λ={lam:.2f}  poly={pm:.4f}  lowess={lm:.4f}  "
              f"Δ={lm-pm:+.4f}  exkurt={avg_stats['exkurt']:.2f}  "
              f"skew={avg_stats['skew']:.2f}")
 
    all_results[reg_name] = reg_res
    print()
 
print("Building plots …")
 
 
# ── plotting ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor("#f8f9fa")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)
 
reg_list = list(REGIMES.keys())
 
for col, reg_name in enumerate(reg_list):
    res = all_results[reg_name]
    lams = LAM_GRID
 
    poly_means  = np.array([np.mean(res[l]["poly"])   for l in lams])
    poly_stds   = np.array([np.std(res[l]["poly"])    for l in lams])
    low_means   = np.array([np.mean(res[l]["lowess"]) for l in lams])
    low_stds    = np.array([np.std(res[l]["lowess"])  for l in lams])
    devs        = low_means - poly_means
    exkurts     = np.array([res[l]["stats"]["exkurt"] for l in lams])
    skews       = np.array([res[l]["stats"]["skew"]   for l in lams])
 
    # ── row 0: absolute prices ────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, col])
    ax0.set_facecolor("white")
    ax0.plot(lams, poly_means, "o-", color="#e74c3c", lw=2, ms=6, label="Polynomial (deg 2)")
    ax0.fill_between(lams, poly_means - poly_stds, poly_means + poly_stds,
                     color="#e74c3c", alpha=0.15)
    ax0.plot(lams, low_means, "s-", color="#2c7bb6", lw=2, ms=6, label=f"LOWESS (frac={FRAC})")
    ax0.fill_between(lams, low_means - low_stds, low_means + low_stds,
                     color="#2c7bb6", alpha=0.15)
    ax0.set_xlabel("Jump Intensity λ (jumps/yr)", fontsize=10)
    ax0.set_ylabel("Put Price", fontsize=10)
    ax0.set_title(f"Price vs λ\n{reg_name}", fontsize=10, fontweight="bold")
    ax0.legend(fontsize=8, framealpha=0.7)
    ax0.grid(True, alpha=0.3, lw=0.5)
 
    # ── row 1: deviation LOWESS − poly with kurtosis overlay ─────────────────
    ax1 = fig.add_subplot(gs[1, col])
    ax1.set_facecolor("white")
 
    bar_colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in devs]
    ax1.bar(lams, devs, width=0.06, color=bar_colors, alpha=0.75,
            edgecolor="white", lw=0.8, label="LOWESS − Poly")
    ax1.errorbar(lams, devs, yerr=np.sqrt(low_stds**2 + poly_stds**2),
                 fmt="none", color="#2c3e50", capsize=4, lw=1.2, alpha=0.6)
    ax1.axhline(0, color="black", lw=1.0)
 
    ax1b = ax1.twinx()
    ax1b.plot(lams, exkurts, "D--", color="#8e44ad", lw=1.5, ms=5,
              alpha=0.8, label="Excess Kurtosis")
    ax1b.set_ylabel("Excess Kurtosis", fontsize=9, color="#8e44ad")
    ax1b.tick_params(axis="y", colors="#8e44ad")
 
    ax1.set_xlabel("Jump Intensity λ", fontsize=10)
    ax1.set_ylabel("Price Deviation (LOWESS − Poly)", fontsize=10)
    ax1.set_title(f"Pricing Gap & Tail Fatness\n{reg_name}", fontsize=10, fontweight="bold")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=7.5, framealpha=0.7)
    ax1.grid(True, alpha=0.3, lw=0.5, axis="y")
 
    # ── row 2: skewness & std of terminal distribution ────────────────────────
    ax2 = fig.add_subplot(gs[2, col])
    ax2.set_facecolor("white")
    term_stds = np.array([res[l]["stats"]["std"] for l in lams])
 
    ax2.plot(lams, skews, "o-", color="#e67e22", lw=2, ms=5, label="Skewness")
    ax2b = ax2.twinx()
    ax2b.plot(lams, term_stds, "s--", color="#27ae60", lw=2, ms=5,
              label="Terminal Std Dev")
    ax2b.set_ylabel("Terminal Std Dev", fontsize=9, color="#27ae60")
    ax2b.tick_params(axis="y", colors="#27ae60")
 
    ax2.axhline(0, color="grey", lw=0.8, ls=":")
    ax2.set_xlabel("Jump Intensity λ", fontsize=10)
    ax2.set_ylabel("Skewness", fontsize=9, color="#e67e22")
    ax2.tick_params(axis="y", colors="#e67e22")
    ax2.set_title(f"Terminal Distribution Shape\n{reg_name}", fontsize=10, fontweight="bold")
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=7.5, framealpha=0.7)
    ax2.grid(True, alpha=0.3, lw=0.5)
 
fig.suptitle(
    "Jump Intensity (λ) Sweep: LOWESS vs Polynomial LSM\n"
    f"M={M:,} paths, {N_SEEDS} seeds averaged, T=1yr, K=100, σ=0.20, LOWESS frac={FRAC}",
    fontsize=13, fontweight="bold", y=1.01
)
 
plt.savefig("jump_intensity_sweep.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved → jump_intensity_sweep.png")