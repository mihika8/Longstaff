import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

def simulate_merton_jump_diffusion_paths(
    S0, r, sigma, T, N, M,
    lam, m_j, v_j,  # jump intensity λ, mean m_j and variance v_j of log-jump
    seed=None
):
    """
    Simulate stock paths under Merton jump–diffusion.
    - dS/S = (r - λ κ) dt + σ dW + (J-1) dN
    - ln J ~ N(m_j, v_j), so κ = E[J - 1] = exp(m_j + 0.5*v_j) - 1
    Returns array of shape (N+1, M): time along axis 0, paths along axis 1.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    kappa = np.exp(m_j + 0.5 * v_j) - 1.0  # mean relative jump size

    # Pre‑allocate paths
    paths = np.zeros((N + 1, M), dtype=float)
    paths[0, :] = S0

    # Random draws
    Z = np.random.normal(size=(N, M))             # Brownian shocks
    K = np.random.poisson(lam * dt, size=(N, M))  # number of jumps per step

    # For jump sizes, we need sum of K normals per (step, path)
    # Efficient approach: generate max_K normals and mask.
    max_K = np.max(K)
    if max_K > 0:
        Y = np.random.normal(loc=m_j, scale=np.sqrt(v_j),
                             size=(max_K, N, M))
    else:
        Y = None

    for t in range(1, N + 1):
        S_prev = paths[t - 1, :]

        # diffusion part (GBM with drift adjusted by jump compensator)
        drift = (r - lam * kappa - 0.5 * sigma**2) * dt
        diff  = sigma * np.sqrt(dt) * Z[t - 1, :]
        S_diff = S_prev * np.exp(drift + diff)

        # jump part: product of exp(Y_i) for i=1..K
        K_t = K[t - 1, :]
        if max_K > 0:
            # sum Y_i for each path where K_t>0
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

def continuation_lowess(S_itm, Y_itm, frac=0.25, it=0):
    order = np.argsort(S_itm)
    x = S_itm[order]
    y = Y_itm[order]
    yhat_sorted = lowess(endog=y, exog=x, frac=frac, it=it, return_sorted=False)
    yhat = np.empty_like(yhat_sorted)
    yhat[order] = yhat_sorted
    return yhat

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
    cf = np.maximum(K - S_T, 0.0)

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

# Example run
S0 = 100
r = 0.02
sigma = 0.2
T = 3
N = 252 * T
M = 20000
lam = 0.2
m_j = -0.5 * 0.04
v_j = 0.04
K = 100

paths = simulate_merton_jump_diffusion_paths(S0, r, sigma, T, N, M, lam, m_j, v_j, seed=42)

price_poly = lsm_price(paths, r, T, K, method="poly")
price_lowess = lsm_price(paths, r, T, K, method="lowess", lowess_frac=0.25)

print("Poly LSM price:", price_poly)
print("LOWESS LSM price:", price_lowess)