import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from statsmodels.nonparametric.smoothers_lowess import lowess


# =========================================================
# 1. Path simulation
# =========================================================

def simulate_merton_jump_diffusion_paths(
    S0, r, sigma, T, N, M,
    lam, m_j=None, v_j=0.0,
    zero_mean_jump=False,
    seed=None
):
    """
    Simulate stock paths under Merton jump-diffusion.

    Model:
        dS/S = (r - lambda*kappa) dt + sigma dW + (J - 1) dN
        ln J ~ N(m_j, v_j)

    If zero_mean_jump=True, choose m_j so that E[J] = 1.
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
        Y = np.random.normal(
            loc=m_j,
            scale=np.sqrt(v_j),
            size=(max_K, N, M)
        )
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


# =========================================================
# 2. European put pricing
# =========================================================

def black_scholes_put(S0, K, r, sigma, T):
    """
    Closed-form European put price under GBM.
    """
    if T <= 0:
        return max(K - S0, 0.0)

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    put = K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return put


def european_put_mc(paths, K, r, T):
    """
    Monte Carlo European put price from terminal payoff.
    Useful for jump-diffusion where we are not using a closed form here.
    """
    payoff = np.maximum(K - paths[-1, :], 0.0)
    disc = math.exp(-r * T)
    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / np.sqrt(len(payoff))
    return price, stderr


# =========================================================
# 3. LOWESS helpers
# =========================================================

def lowess_fit_same_x(x, y, frac):
    """
    Fit LOWESS and return fitted values at the same x points.
    """
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

    yhat = np.empty_like(yhat_sorted)
    yhat[order] = yhat_sorted
    return yhat


def approximate_gcv_score(x, y, frac):
    """
    Approximate GCV-style score for LOWESS.

    Since statsmodels.lowess does not expose the smoother matrix / EDF,
    we use a practical proxy:
        score = RSS / (n * (1 - k/n)^2)
    where k ~ frac * n is a rough complexity proxy.

    This is not exact LOESS GCV, but it gives an automated span-selection
    rule that is far better than fixing frac arbitrarily.
    """
    n = len(x)
    if n < 10:
        return np.inf

    yhat = lowess_fit_same_x(x, y, frac)
    rss = np.sum((y - yhat) ** 2)

    k = max(2, int(np.ceil(frac * n)))
    penalty = (1.0 - k / n) ** 2

    if penalty <= 1e-12:
        return np.inf

    return rss / (n * penalty)


def select_lowess_frac_gcv(x, y, frac_grid=None):
    """
    Search over candidate LOWESS spans and pick the one with the smallest
    approximate GCV score.
    """
    if frac_grid is None:
        frac_grid = np.linspace(0.10, 0.60, 11)

    best_frac = None
    best_score = np.inf
    best_fit = None

    for frac in frac_grid:
        score = approximate_gcv_score(x, y, frac)
        if score < best_score:
            best_score = score
            best_frac = frac
            best_fit = lowess_fit_same_x(x, y, frac)

    return best_frac, best_score, best_fit


# =========================================================
# 4. LSM American put pricing
# =========================================================

def american_put_lsm(
    paths, K, r, T,
    method="poly",
    degree=2,
    frac=0.25,
    frac_grid=None,
    min_itm=25
):
    """
    Price an American put by LSM.

    method:
        - 'poly'
        - 'lowess_fixed'
        - 'lowess_gcv'

    Returns:
        dict with price, stderr, average chosen frac, and exercise boundary.
    """
    n_steps = paths.shape[0] - 1
    dt = T / n_steps
    disc = math.exp(-r * dt)

    cashflows = np.maximum(K - paths[-1], 0.0)

    exercise_time = np.full(paths.shape[1], n_steps, dtype=int)
    exercised_early = np.zeros(paths.shape[1], dtype=bool)

    chosen_fracs = []
    boundary = []

    for t in range(n_steps - 1, 0, -1):
        S_t = paths[t, :]
        exercise_val = np.maximum(K - S_t, 0.0)
        itm = exercise_val > 0

        cashflows *= disc

        if not np.any(itm):
            boundary.append((t, np.nan))
            continue

        x = S_t[itm]
        y = cashflows[itm]

        if method == "poly" or len(x) < min_itm:
            X = np.vander(x, N=degree + 1, increasing=True)
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            cont = X @ beta
            used_frac = np.nan

        elif method == "lowess_fixed":
            cont = lowess_fit_same_x(x, y, frac)
            used_frac = frac
            chosen_fracs.append(frac)

        elif method == "lowess_gcv":
            best_frac, best_score, cont = select_lowess_frac_gcv(x, y, frac_grid=frac_grid)
            used_frac = best_frac
            chosen_fracs.append(best_frac)

        else:
            raise ValueError("Unknown method")

        ex_now = exercise_val[itm] > cont
        itm_idx = np.where(itm)[0]
        ex_idx = itm_idx[ex_now]

        if len(ex_idx) > 0:
            cashflows[ex_idx] = exercise_val[ex_idx]
            exercise_time[ex_idx] = t
            exercised_early[ex_idx] = True
            boundary.append((t, np.max(S_t[ex_idx])))
        else:
            boundary.append((t, np.nan))

    price = disc * cashflows.mean()
    stderr = disc * cashflows.std(ddof=1) / np.sqrt(len(cashflows))

    boundary_df = pd.DataFrame(boundary, columns=["time_index", "boundary_stock"])
    boundary_df["time"] = boundary_df["time_index"] * dt
    boundary_df = boundary_df.sort_values("time")

    return {
        "price": price,
        "stderr": stderr,
        "avg_frac": np.mean(chosen_fracs) if len(chosen_fracs) > 0 else np.nan,
        "fracs_used": chosen_fracs,
        "exercise_boundary": boundary_df,
        "early_exercise_rate": exercised_early.mean()
    }


# =========================================================
# 5. Terminal distribution summary
# =========================================================

def summarize_terminal_distribution(paths):
    ST = paths[-1, :]
    return {
        "terminal_mean": ST.mean(),
        "terminal_std": ST.std(ddof=1),
        "terminal_q05": np.quantile(ST, 0.05),
        "terminal_q50": np.quantile(ST, 0.50),
        "terminal_q95": np.quantile(ST, 0.95)
    }


# =========================================================
# 6. Single-scenario runner
# =========================================================

def run_scenario(
    label,
    model_name,
    lam,
    regression,
    S0=40,
    K=40,
    r=0.06,
    sigma=0.20,
    T=1.0,
    N=50,
    M=5000,
    v_j=0.04,
    degree=2,
    fixed_frac=0.25,
    frac_grid=None,
    seed=123
):
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
        seed=seed
    )

    if regression == "polynomial":
        amer = american_put_lsm(
            paths, K, r, T,
            method="poly",
            degree=degree
        )

    elif regression == "LOWESS_fixed":
        amer = american_put_lsm(
            paths, K, r, T,
            method="lowess_fixed",
            frac=fixed_frac
        )

    elif regression == "LOWESS_GCV":
        amer = american_put_lsm(
            paths, K, r, T,
            method="lowess_gcv",
            frac_grid=frac_grid
        )

    else:
        raise ValueError("Unknown regression")

    if model_name == "GBM":
        euro = black_scholes_put(S0, K, r, sigma, T)
        euro_se = np.nan
    else:
        euro, euro_se = european_put_mc(paths, K, r, T)

    eep = amer["price"] - euro
    dist = summarize_terminal_distribution(paths)

    result = {
        "Scenario": label,
        "Model": model_name,
        "lambda": lam,
        "Regression": regression,
        "AmericanPut": amer["price"],
        "AmericanSE": amer["stderr"],
        "EuropeanPut": euro,
        "EuropeanSE": euro_se,
        "EarlyExercisePremium": eep,
        "EarlyExerciseRate": amer["early_exercise_rate"],
        "AvgLOWESSFrac": amer["avg_frac"],
        **dist
    }

    return result, amer["exercise_boundary"]


# =========================================================
# 7. Main experiment
# =========================================================

def main():
    S0 = 40
    K = 40
    r = 0.06
    sigma = 0.20
    T = 1.0
    N = 50
    M = 5000
    v_j = 0.04

    frac_grid = np.linspace(0.10, 0.60, 11)

    scenarios = [
        ("A1", "GBM", 0.00, "polynomial", 101),
        ("A2", "Jump-diffusion", 0.05, "polynomial", 102),
        ("B1", "Jump-diffusion", 0.05, "polynomial", 103),
        ("B2", "Jump-diffusion", 0.05, "LOWESS_GCV", 103),
    ]

    results = []
    boundaries = {}

    for label, model_name, lam, regression, seed in scenarios:
        res, boundary = run_scenario(
            label=label,
            model_name=model_name,
            lam=lam,
            regression=regression,
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            N=N,
            M=M,
            v_j=v_j,
            degree=2,
            fixed_frac=0.25,
            frac_grid=frac_grid,
            seed=seed
        )
        results.append(res)
        boundaries[label] = boundary

    df = pd.DataFrame(results)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")

    print("\n===== EXPERIMENT RESULTS =====\n")
    print(df)

    print("\n===== INTERPRETATION =====\n")
    print("A1 vs A2: isolates the effect of introducing jumps while keeping polynomial regression fixed.")
    print("B1 vs B2: isolates the effect of replacing polynomial regression with LOWESS_GCV under jump-diffusion.")

    if "A1" in df["Scenario"].values and "A2" in df["Scenario"].values:
        a1 = df[df["Scenario"] == "A1"].iloc[0]
        a2 = df[df["Scenario"] == "A2"].iloc[0]
        print(f"\nA1 -> A2 American put change: {a2['AmericanPut'] - a1['AmericanPut']:.4f}")
        print(f"A1 -> A2 EEP change:          {a2['EarlyExercisePremium'] - a1['EarlyExercisePremium']:.4f}")

    if "B1" in df["Scenario"].values and "B2" in df["Scenario"].values:
        b1 = df[df["Scenario"] == "B1"].iloc[0]
        b2 = df[df["Scenario"] == "B2"].iloc[0]
        print(f"\nB1 -> B2 American put change: {b2['AmericanPut'] - b1['AmericanPut']:.4f}")
        print(f"B2 selected avg LOWESS frac:  {b2['AvgLOWESSFrac']:.4f}")

    print("\n===== SAMPLE EXERCISE BOUNDARIES =====\n")
    for key, bd in boundaries.items():
        print(f"\nBoundary for {key}:")
        print(bd.head(10))

    df.to_csv("experiment_results.csv", index=False)
    print("\nSaved results to experiment_results.csv")


if __name__ == "__main__":
    main()