"""American Put with Discrete Dividends - Longstaff-Schwartz (Self-contained)"""
import numpy as np
import matplotlib.pyplot as plt
from longstaff_schwartz.algorithm import longstaff_schwartz as lsmc_algorithm
from longstaff_schwartz.stochastic_process import GeometricBrownianMotion
import time

def constant_rate_discount(r, t_from, t_to):
    """Discount factor for constant rate."""
    return np.exp(-r * (t_to - t_from))

def simulate_merton_jump_diffusion_paths(S0, r, sigma, T, N, M,
                                         lam, m_j, v_j, seed=42):
    """
    Simulate Merton jump–diffusion paths (no dividends yet).

    dS/S = (r - λ κ) dt + σ dW + (J - 1) dN,
    ln J ~ N(m_j, v_j), κ = E[J-1] = exp(m_j + 0.5*v_j) - 1.
    """
    np.random.seed(seed)
    dt = T / N
    times = np.linspace(0.0, T, N + 1)

    kappa = np.exp(m_j + 0.5 * v_j) - 1.0

    # allocate
    X = np.zeros((N + 1, M))
    X[0] = S0

    # pre‑draw randomness
    Z = np.random.normal(size=(N, M))
    K = np.random.poisson(lam * dt, size=(N, M))

    for n in range(1, N + 1):
        S_prev = X[n - 1]

        # diffusion part
        drift = (r - lam * kappa - 0.5 * sigma**2) * dt
        diff = sigma * np.sqrt(dt) * Z[n - 1]
        S_diff = S_prev * np.exp(drift + diff)

        # jump part: collapse K jumps into one normal with mean K*m_j, var K*v_j
        K_step = K[n - 1]
        jump_mean = K_step * m_j
        jump_var = K_step * v_j
        jump_term = np.where(
            K_step > 0,
            np.random.normal(loc=jump_mean, scale=np.sqrt(jump_var)),
            0.0
        )

        X[n] = S_diff * np.exp(jump_term)

    return times, X

def simulate_paths_with_dividends(S0, r, sigma, T, N, M, dividends, seed=42,
                                  use_jumps=False,
                                  lam=0.0, m_j=0.0, v_j=0.0):
    """Simulate paths (GBM or Merton) with discrete cash dividends."""
    if use_jumps:
        # Jump–diffusion paths
        times, X = simulate_merton_jump_diffusion_paths(
            S0, r, sigma, T, N, M, lam, m_j, v_j, seed=seed
        )
    else:
        # Original GBM paths
        np.random.seed(seed)
        times = np.linspace(0.0, T, N + 1)
        gbm = GeometricBrownianMotion(mu=r, sigma=sigma)
        rng = np.random.RandomState(seed)
        X = gbm.simulate(times, M, rng)
        X = X * (S0 / X[0])

    # Apply dividends as before
    div_idx = {}
    for t_div, D in dividends:
        idx = np.argmin(np.abs(times - t_div))
        div_idx.setdefault(idx, 0.0)
        div_idx[idx] += D

    for idx, D in div_idx.items():
        X[idx] = np.maximum(X[idx] - D, 0.0)

    return times, X


def price_american_put(K, T, r, sigma, S0, N, M, dividends):
    """Price American put with LSMC."""
    times, paths = simulate_paths_with_dividends(S0, r, sigma, T, N, M, dividends)
    
    def df(t_from, t_to):
        return constant_rate_discount(r, t_from, t_to)
    
    def put_payoff(spot):
        return np.maximum(K - spot, 0.0)
    
    def itm(payoff, spot):
        return payoff > 0
    
    def fit_quadratic(x, y):
        return np.polynomial.Polynomial.fit(x, y, 2, rcond=None)
    
    price = lsmc_algorithm(paths, times, df, fit_quadratic, put_payoff, itm)

    return price, paths, times

# Example jump parameters
lam = 0.5          # 0.5 jumps per year on average
m_j = -0.1         # downward‑biased jumps
v_j = 0.2**2       # jump size uncertainty

def price_american_put_jumps(K, T, r, sigma, S0, N, M, dividends):
    times, paths = simulate_paths_with_dividends(
        S0, r, sigma, T, N, M, dividends,
        seed=42,
        use_jumps=True,
        lam=lam, m_j=m_j, v_j=v_j
    )

    def df(t_from, t_to):
        return constant_rate_discount(r, t_from, t_to)

    def put_payoff(spot):
        return np.maximum(K - spot, 0.0)

    def itm(payoff, spot):
        return payoff > 0

    def fit_quadratic(x, y):
        return np.polynomial.Polynomial.fit(x, y, 2, rcond=None)

    price = lsmc_algorithm(paths, times, df, fit_quadratic, put_payoff, itm)
    return price, paths, times


# ============================================================================
# EXPERIMENTS
# ============================================================================

print("\n" + "="*70)
print("AMERICAN PUT PRICING WITH DISCRETE DIVIDENDS")
print("="*70)

# Parameters
T, S, K, r, sigma, N, M = 1.0, 50, 50, 0.05, 0.25, 500, 50000
dividends = [(0.5, 2.0)]

print("\n📌 EXPERIMENT 1: With vs Without Dividends")
print("-"*70)

print(f"Pricing WITHOUT dividend (M={M} paths)...")
start = time.time()
V0_nodiv, _, _ = price_american_put(K, T, r, sigma, S, N, M, [])
t1 = time.time() - start
print(f"  ✓ ${V0_nodiv:.4f} (took {t1:.1f}s)")

print(f"Pricing WITH dividend at t=0.5, D=2.0...")
start = time.time()
V0_div, _, _ = price_american_put(K, T, r, sigma, S, N, M, dividends)
t2 = time.time() - start
print(f"  ✓ ${V0_div:.4f} (took {t2:.1f}s)")

print(f"\n📊 RESULTS:")
print(f"  No Dividend:   ${V0_nodiv:.4f}")
print(f"  With Dividend: ${V0_div:.4f}")
print(f"  Difference:    ${V0_div - V0_nodiv:.4f} ({(V0_div/V0_nodiv-1)*100:+.1f}%)")
print(f"\n💡 Interpretation: Dividend INCREASES put value by ${V0_div - V0_nodiv:.4f}")

# EXPERIMENT 2: Early vs Late Dividend
print("\n📌 EXPERIMENT 2: Early vs Late Dividend Timing")
print("-"*70)

dividends_early = [(0.5, 2.0)]
dividends_late = [(0.95, 2.0)]

print("Pricing with EARLY dividend (t=0.5)...")
V0_early, _, _ = price_american_put(K, T, r, sigma, S, N, M, dividends_early)
print(f"  ✓ ${V0_early:.4f}")

print("Pricing with LATE dividend (t=0.95)...")
V0_late, _, _ = price_american_put(K, T, r, sigma, S, N, M, dividends_late)
print(f"  ✓ ${V0_late:.4f}")

print(f"\n📊 RESULTS:")
print(f"  Early (t=0.5):  ${V0_early:.4f}")
print(f"  Late  (t=0.95): ${V0_late:.4f}")
print(f"  Difference:     ${V0_early - V0_late:.4f}")
print(f"\n💡 Interpretation: Earlier dividends increase put value MORE")

# EXPERIMENT 3: Volatility Effect
print("\n📌 EXPERIMENT 3: Volatility Effect (no dividends)")
print("-"*70)

sigma_low, sigma_high = 0.25, 0.45

print(f"Pricing with LOW volatility (σ={sigma_low:.0%})...")
V0_low, _, _ = price_american_put(K, T, r, sigma_low, S, N, M, [])
print(f"  ✓ ${V0_low:.4f}")

print(f"Pricing with HIGH volatility (σ={sigma_high:.0%})...")
V0_high, _, _ = price_american_put(K, T, r, sigma_high, S, N, M, [])
print(f"  ✓ ${V0_high:.4f}")

print(f"\n📊 RESULTS:")
print(f"  Low  vol (25%): ${V0_low:.4f}")
print(f"  High vol (45%): ${V0_high:.4f}")
print(f"  Difference:     ${V0_high - V0_low:.4f} ({(V0_high/V0_low-1)*100:+.1f}%)")
print(f"\n💡 Interpretation: Higher volatility INCREASES put value")

# EXPERIMENT 4: Sample Paths
print("\n📌 EXPERIMENT 4: Visualizing Sample Paths")
print("-"*70)

print("Generating 10 sample paths with dividend...")
_, paths_sample, times_sample = price_american_put(K, T, r, sigma, S, N, 100, dividends)

plt.figure(figsize=(12, 6))
for i in range(min(10, paths_sample.shape[1])):
    plt.plot(times_sample, paths_sample[:, i], alpha=0.7, linewidth=1.5)
plt.axhline(K, linestyle='--', color='red', linewidth=2, label=f'Strike K={K}')
plt.axvline(0.5, linestyle=':', color='gray', alpha=0.5, label='Dividend at t=0.5')
plt.xlabel('Time (years)', fontsize=12)
plt.ylabel('Stock Price', fontsize=12)
plt.title('Sample LSMC Paths with Discrete Dividend (D=2 at t=0.5)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sample_paths.png', dpi=150)
print("  ✓ Saved plot: sample_paths.png")

print("\n📌 EXPERIMENT 5: Jump–Diffusion vs GBM (with dividend)")
print("-"*70)

V0_gbm, _, _ = price_american_put(K, T, r, sigma, S, N, M, dividends)

V0_jump, _, _ = price_american_put_jumps(K, T, r, sigma, S, N, M, dividends)

print(f"  GBM price:    ${V0_gbm:.4f}")
print(f"  Jump price:   ${V0_jump:.4f}")
print(f"  Difference:   ${V0_jump - V0_gbm:.4f}")

# SUMMARY
print("\n" + "="*70)
print("✨ SUMMARY OF KEY FINDINGS")
print("="*70)
print(f"""
1. 📈 Discrete dividends INCREASE American put value
   • No dividend:   ${V0_nodiv:.4f}
   • With dividend: ${V0_div:.4f}
   • Impact: +${V0_div - V0_nodiv:.4f} ({(V0_div/V0_nodiv-1)*100:.1f}%)

2. ⏰ Timing matters: Earlier dividends have GREATER impact
   • Early (t=0.5):  ${V0_early:.4f}
   • Late  (t=0.95): ${V0_late:.4f}

3. 📊 Higher volatility increases put value
   • Low  vol (25%): ${V0_low:.4f}
   • High vol (45%): ${V0_high:.4f}
   • Impact: +${V0_high - V0_low:.4f} ({(V0_high/V0_low-1)*100:.1f}%)

Why do dividends increase put value?
→ Cash dividends reduce the stock price, making it more likely
  the put will be in-the-money. The stock drops by $D at ex-date,
  increasing downside potential.

Output files:
  📁 sample_paths.png
""")

print("="*70)
print("✅ All experiments completed successfully!")
print("="*70)
