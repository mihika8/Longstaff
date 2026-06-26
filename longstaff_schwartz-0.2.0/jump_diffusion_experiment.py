import numpy as np
import pandas as pd
import math
import os
from scipy.stats import skew, kurtosis
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt


def simulate_merton_jump_diffusion_paths(
    S0, r, sigma, T, N, M,
    lam, m_j=None, v_j=0.0,
    zero_mean_jump=False,
    seed=None
):
    """
    Simulate Merton jump-diffusion stock paths.

    dS/S = (r - lambda*kappa)dt + sigma dW + (J - 1)dN
    ln J ~ N(m_j, v_j)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N

    if zero_mean_jump:
        m_j = -0.5 * v_j
    elif m_j is None:
        raise ValueError("Provide m_j unless zero_mean_jump=True")

    kappa = np.exp(m_j + 0.5 * v_j) - 1.0

    paths = np.zeros((N + 1, M), dtype=float)
    paths[0, :] = S0

    Z = np.random.normal(size=(N, M))
    K = np.random.poisson(lam * dt, size=(N, M))

    max_K = np.max(K)
    if max_K > 0:
        Y = np.random.normal(loc=m_j, scale=np.sqrt(v_j), size=(max_K, N, M))
    else:
        Y = None

    drift = (r - lam * kappa - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    for t in range(1, N + 1):
        S_prev = paths[t - 1, :]
        S_diff = S_prev * np.exp(drift + vol * Z[t - 1, :])

        K_t = K[t - 1, :]
        if max_K > 0:
            jump_sum = np.zeros(M)
            for k in range(1, max_K + 1):
                mask = (K_t >= k)
                if np.any(mask):
                    jump_sum[mask] += Y[k - 1, t - 1, mask]
            paths[t, :] = S_diff * np.exp(jump_sum)
        else:
            paths[t, :] = S_diff

    return paths


def american_put_lsm(paths, K, r, T, method="poly", frac=0.25, degree=2):
    """
    Price an American put by Least-Squares Monte Carlo.
    method = 'poly' or 'lowess'
    """
    n_steps = paths.shape[0] - 1
    dt = T / n_steps
    disc = math.exp(-r * dt)

    cashflows = np.maximum(K - paths[-1], 0.0)

    for t in range(n_steps - 1, 0, -1):
        S_t = paths[t]
        exercise = np.maximum(K - S_t, 0.0)
        itm = exercise > 0

        cashflows *= disc

        if np.any(itm):
            x = S_t[itm]
            y = cashflows[itm]

            if method == "lowess" and len(x) >= 25:
                order = np.argsort(x)
                x_sorted = x[order]
                y_sorted = y[order]

                yhat_sorted = lowess(
                    endog=y_sorted,
                    exog=x_sorted,
                    frac=frac,
                    it=0,
                    return_sorted=False
                )

                cont = np.empty_like(yhat_sorted)
                cont[order] = yhat_sorted
            else:
                X = np.vander(x, N=degree + 1, increasing=True)
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                cont = X @ beta

            ex_now = exercise[itm] > cont
            idx = np.where(itm)[0]
            ex_idx = idx[ex_now]
            cashflows[ex_idx] = exercise[ex_idx]

    price = disc * cashflows.mean()
    stderr = disc * cashflows.std(ddof=1) / np.sqrt(len(cashflows))
    return price, stderr


def summarize_terminal_distribution(paths):
    ST = paths[-1, :]
    return {
        "terminal_mean": ST.mean(),
        "terminal_std": ST.std(ddof=1),
        "terminal_skew": skew(ST),
        "terminal_kurtosis": kurtosis(ST, fisher=False),
        "q05": np.quantile(ST, 0.05),
        "q50": np.quantile(ST, 0.50),
        "q95": np.quantile(ST, 0.95),
    }


def run_experiment():
    os.makedirs("output", exist_ok=True)

    S0 = 100
    Kstrike = 100
    r = 0.05
    sigma = 0.20
    M = 5000

    lam_grid = [0.05, 0.20, 0.50]
    T_grid = [1, 3, 5]
    v_j = 0.04
    seed_base = 123

    rows = []

    for i, lam in enumerate(lam_grid):
        for j, T in enumerate(T_grid):
            N = int(100 * T)

            paths = simulate_merton_jump_diffusion_paths(
                S0=S0,
                r=r,
                sigma=sigma,
                T=T,
                N=N,
                M=M,
                lam=lam,
                v_j=v_j,
                zero_mean_jump=True,
                seed=seed_base + i * 10 + j
            )

            dist = summarize_terminal_distribution(paths)

            poly_price, poly_se = american_put_lsm(
                paths, Kstrike, r, T, method="poly", degree=2
            )

            lowess_price, lowess_se = american_put_lsm(
                paths, Kstrike, r, T, method="lowess", frac=0.25
            )

            rows.append({
                "lambda": lam,
                "T": T,
                **dist,
                "poly_price": poly_price,
                "poly_se": poly_se,
                "lowess_price": lowess_price,
                "lowess_se": lowess_se,
                "price_diff_lowess_minus_poly": lowess_price - poly_price
            })

    res = pd.DataFrame(rows).sort_values(["lambda", "T"]).reset_index(drop=True)

    res.to_csv("output/jump_diffusion_lsm_results.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for lam in lam_grid:
        sub = res[res["lambda"] == lam]
        axes[0].plot(sub["T"], sub["poly_price"], marker="o", label=f"Poly λ={lam}")
        axes[0].plot(sub["T"], sub["lowess_price"], marker="s", linestyle="--", label=f"LOWESS λ={lam}")

    axes[0].set_title("American put price by maturity")
    axes[0].set_xlabel("Maturity T (years)")
    axes[0].set_ylabel("Estimated price")
    axes[0].legend(fontsize=8)

    for lam in lam_grid:
        sub = res[res["lambda"] == lam]
        axes[1].plot(sub["T"], sub["terminal_std"], marker="o", label=f"λ={lam}")

    axes[1].set_title("Terminal price dispersion by maturity")
    axes[1].set_xlabel("Maturity T (years)")
    axes[1].set_ylabel("Std. dev. of $S_T$")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("output/jump_diffusion_lsm_results.png", dpi=200, bbox_inches="tight")

    print("\n===== RESULTS TABLE =====\n")
    print(res.round(4).to_string(index=False))
    print("\nSaved:")
    print(" - output/jump_diffusion_lsm_results.csv")
    print(" - output/jump_diffusion_lsm_results.png")


if __name__ == "__main__":
    run_experiment()