import numpy as np

def simulate_merton_jump_diffusion_paths(
    S0, r, sigma, T, N, M,
    lam, m_j=None, v_j=0.0,
    zero_mean_jump=False,
    seed=None
):
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

lam_grid = [0.05, 0.10, 0.20, 0.50]
T_grid = [1, 2, 3, 5]

from scipy.stats import skew, kurtosis

def summarize_terminal_distribution(paths):
    ST = paths[-1, :]
    return {
        "mean": ST.mean(),
        "std": ST.std(ddof=1),
        "skew": skew(ST),
        "kurtosis": kurtosis(ST, fisher=False),
        "q05": np.quantile(ST, 0.05),
        "q50": np.quantile(ST, 0.50),
        "q95": np.quantile(ST, 0.95),
    }

def run_scenarios(S0, r, sigma, M, lam_grid, T_grid, v_j, seed=42):
    results = {}

    for lam in lam_grid:
        for T in T_grid:
            N = int(252 * T)
            paths = simulate_merton_jump_diffusion_paths(
                S0=S0, r=r, sigma=sigma, T=T, N=N, M=M,
                lam=lam, v_j=v_j, zero_mean_jump=True, seed=seed
            )
            results[(lam, T)] = {
                "paths": paths,
                "dist_summary": summarize_terminal_distribution(paths)
            }

    return results

if __name__ == "__main__":
    # Example parameters
    S0 = 100.0
    r = 0.02
    sigma = 0.2
    M = 10000
    v_j = 0.04  # e.g. std of log-jump = 0.2

    results = run_scenarios(
        S0=S0,
        r=r,
        sigma=sigma,
        M=M,
        lam_grid=lam_grid,
        T_grid=T_grid,
        v_j=v_j,
        seed=42
    )

    for (lam, T), res in results.items():
        print(f"lam={lam}, T={T}, summary={res['dist_summary']}")