import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess


# ---------- Path simulation: Merton jump-diffusion ----------

def simulate_merton_jump_diffusion_paths(
    S0, r, sigma, T, N, M,
    lam, m_j, v_j,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
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
                if not np.any(mask):
                    continue
                jump_sum[mask] += Y[k - 1, t - 1, mask]
            S_t = S_diff * np.exp(jump_sum)
        else:
            S_t = S_diff

        paths[t, :] = S_t

    return paths


# ---------- LOWESS helper ----------

def continuation_lowess(S_itm, Y_itm, frac=0.25, it=0):
    order = np.argsort(S_itm)
    x = np.asarray(S_itm[order], dtype=float)
    y = np.asarray(Y_itm[order], dtype=float)

    yhat_sorted = lowess(
        endog=y,
        exog=x,
        frac=frac,
        it=it,
        return_sorted=False
    )

    yhat = np.empty_like(yhat_sorted)
    yhat[order] = yhat_sorted
    return yhat


# ---------- LSM pricer (poly or LOWESS) ----------

def lsm_price(
    paths,
    r,
    T,
    K,
    method="poly",
    poly_degree=2,
    lowess_frac=0.25,
    lowess_it=0
):
    N, M = paths.shape[0] - 1, paths.shape[1]
    dt = T / N
    disc = np.exp(-r * dt)

    S_T = paths[-1, :]
    cf = np.maximum(K - S_T, 0.0)  # American put

    for t in range(N - 1, 0, -1):
        S_t = paths[t, :]
        immediate = np.maximum(K - S_t, 0.0)
        itm = immediate > 0
        if not np.any(itm):
            cf *= disc
            continue

        cont_disc_all = cf * disc
        x = S_t[itm]
        y = cont_disc_all[itm]

        if method == "lowess":
            if len(x) >= 20:
                cont_itm = continuation_lowess(x, y, frac=lowess_frac, it=lowess_it)
            else:
                X = np.column_stack([np.ones_like(x), x])
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                cont_itm = X @ beta
        else:
            X = np.ones((len(x), poly_degree + 1))
            for d in range(1, poly_degree + 1):
                X[:, d] = x**d
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            cont_itm = X @ beta

        cont_all = np.zeros_like(S_t)
        cont_all[itm] = cont_itm

        exercise_now = itm & (immediate > cont_all)

        cf[exercise_now] = immediate[exercise_now]
        cf[~exercise_now] = cont_disc_all[~exercise_now]

    price = np.exp(-r * dt) * cf.mean()
    return price


# ---------- Monte Carlo experiment wrapper ----------

def mc_experiment(
    n_runs,
    method,
    S0, r, sigma, T, N, M,
    lam, m_j, v_j, K,
    **kwargs
):
    prices = []
    for seed in range(n_runs):
        paths = simulate_merton_jump_diffusion_paths(
            S0, r, sigma, T, N, M, lam, m_j, v_j, seed=seed
        )
        price = lsm_price(paths, r, T, K, method=method, **kwargs)
        prices.append(price)
    return np.array(prices)


# ---------- Main: run once + MC comparison ----------

if __name__ == "__main__":
    # Parameters
    S0 = 100
    r = 0.02
    sigma = 0.2
    T = 3
    N = 252 * T
    M = 5000           # keep moderate for speed; increase later if needed
    lam = 0.2
    v_j = 0.04
    m_j = -0.5 * v_j   # zero-mean jump on J scale
    K = 100

    # Single run on one path set
    paths = simulate_merton_jump_diffusion_paths(
        S0, r, sigma, T, N, M, lam, m_j, v_j, seed=42
    )

    price_poly = lsm_price(paths, r, T, K, method="poly", poly_degree=2)
    price_lowess = lsm_price(paths, r, T, K, method="lowess", lowess_frac=0.25)

    print("Single-run prices:")
    print(f"  Poly   LSM price:  {price_poly:.6f}")
    print(f"  LOWESS LSM price:  {price_lowess:.6f}")

    # Monte Carlo comparison
    n_runs = 20  # increase to 50+ if you’re patient
    prices_poly = mc_experiment(
        n_runs, "poly",
        S0, r, sigma, T, N, M, lam, m_j, v_j, K,
        poly_degree=2
    )
    prices_lowess = mc_experiment(
        n_runs, "lowess",
        S0, r, sigma, T, N, M, lam, m_j, v_j, K,
        lowess_frac=0.25
    )

    print("\nMonte Carlo comparison over", n_runs, "runs:")
    print("  Poly   LSM:   mean = {:.6f}, std = {:.6f}".format(
        prices_poly.mean(), prices_poly.std(ddof=1)))
    print("  LOWESS LSM:   mean = {:.6f}, std = {:.6f}".format(
        prices_lowess.mean(), prices_lowess.std(ddof=1)))