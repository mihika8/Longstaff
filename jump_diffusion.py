import numpy as np

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
