import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Ensure absolute simulation reproducibility across runs
np.random.seed(42)

## =====================================================================
## 1. FIXED EXPERIMENT PARAMETERS
## =====================================================================
S0 = 40.0       # Initial stock price
K = 40.0        # Strike price
r = 0.06        # Risk-free rate
sigma = 0.20    # Volatility
T = 1.0         # Time to maturity
N_steps = 50    # Number of exercise dates (50-step grid)
dt = T / N_steps

# Merton Jump-Diffusion Parameters (Scenario A2 / B1 / B2)
lam = 0.05       # Jump intensity (rate of jumps per year)
mu_J = 0.0      # Mean of log jump size
sigma_J = 0.4   # Volatility of log jump size
kappa = np.exp(mu_J + 0.5 * sigma_J**2) - 1

# Simulation Path Size
M_paths = 10000  # Paths used to evaluate the estimators

## =====================================================================
## 2. DETERMINISTIC TREE BENCHMARK WITH BOUNDARY EXTRACTION
## =====================================================================
def merton_tree_benchmark_with_boundary(S0, K, r, sigma, lam, mu_J, sigma_J, T, N_tree_steps=200):
    """
    Prices an American Put using the Amin (1993) Jump-Diffusion Tree framework
    and extracts the exact theoretical early exercise boundary S*.
    """
    dt_t = T / N_tree_steps
    dx = sigma * np.sqrt(dt_t)
    u = np.exp(dx)
    d = np.exp(-dx)
    
    p_base = (np.exp((r - lam * kappa) * dt_t) - d) / (u - d)
    p_base = max(0.0, min(1.0, p_base))
    
    max_jump_nodes = int(5 * (sigma_J / dx)) 
    jump_range = np.arange(-max_jump_nodes, max_jump_nodes + 1)
    
    jump_probs = np.zeros(len(jump_range))
    for idx, k in enumerate(jump_range):
        lower = k * dx - 0.5 * dx
        upper = k * dx + 0.5 * dx
        jump_probs[idx] = stats.norm.cdf(upper, loc=mu_J, scale=sigma_J) - stats.norm.cdf(lower, loc=mu_J, scale=sigma_J)
        
    jump_probs /= np.sum(jump_probs)
    
    grid_size = 2 * N_tree_steps + 1
    V = np.zeros(grid_size)
    S = np.zeros(grid_size)
    center = N_tree_steps
    
    for i in range(grid_size):
        S[i] = S0 * np.exp((i - center) * dx)
        V[i] = max(K - S[i], 0.0)
        
    tree_time_grid = np.arange(N_tree_steps + 1) * dt_t
    tree_boundary = np.zeros(N_tree_steps + 1)
    tree_boundary[-1] = K
    
    for t in range(N_tree_steps - 1, -1, -1):
        V_next = np.zeros(grid_size)
        highest_exercise_S = 0.0
        
        for i in range(center - t, center + t + 1):
            V_no_jump = p_base * V[i + 1] + (1 - p_base) * V[i - 1]
            V_jump_integral = 0.0
            for k_idx, k in enumerate(jump_range):
                target_node = max(0, min(grid_size - 1, i + k))
                V_jump_integral += jump_probs[k_idx] * V[target_node]
                
            prob_jump = lam * dt_t
            continuation = ((1 - prob_jump) * V_no_jump + prob_jump * V_jump_integral) * np.exp(-r * dt_t)
            
            intrinsic = max(K - S[i], 0.0)
            V_next[i] = max(intrinsic, continuation)
            
            if intrinsic > continuation and intrinsic > 0:
                if S[i] > highest_exercise_S:
                    highest_exercise_S = S[i]
                    
        tree_boundary[t] = highest_exercise_S if highest_exercise_S > 0 else np.nan
        V = V_next.copy()
        
    return V[center], tree_time_grid, tree_boundary

## =====================================================================
## 3. MONTE CARLO JUMP-DIFFUSION PATH GENERATION
## =====================================================================
def simulate_merton_paths(S0, r, sigma, lam, mu_J, sigma_J, kappa, T, N_steps, M_paths):
    S = np.zeros((N_steps + 1, M_paths))
    S[0, :] = S0
    for t in range(1, N_steps + 1):
        Z = np.random.standard_normal(M_paths)
        N = np.random.poisson(lam * dt, M_paths)
        log_J = np.random.normal(mu_J, sigma_J, M_paths) * N
        drift = (r - 0.5 * sigma**2 - lam * kappa) * dt
        S[t, :] = S[t-1, :] * np.exp(drift + sigma * np.sqrt(dt) * Z + log_J)
    return S

## =====================================================================
## 4. AMERICAN OPTION PRICING ENGINE WITH BOUNDARY TRACKING
## =====================================================================
def price_lsm_american_with_boundary(S, K, r, dt, method='poly', deg=3, alpha=0.4):
    """Runs the Longstaff-Schwartz algorithm on ITM paths and logs the boundary."""
    N_steps, M = S.shape[0] - 1, S.shape[1]
    payoffs = np.maximum(K - S, 0)
    cashflows = payoffs[-1, :].copy()
    
    boundary_track = np.zeros(N_steps + 1)
    boundary_track[-1] = K
    
    for t in range(N_steps - 1, 0, -1):
        itm = np.where(payoffs[t, :] > 0)[0]
        if len(itm) > 0:
            X = S[t, itm]
            Y = cashflows[itm] * np.exp(-r * dt)
            
            if method == 'poly':
                coef = np.polyfit(X, Y, deg)
                continuation = np.polyval(coef, X)
            elif method == 'lowess':
                continuation = sm.nonparametric.lowess(Y, X, frac=alpha, it=0, return_sorted=False)
                
            exercise = payoffs[t, itm] > continuation
            cashflows[itm[exercise]] = payoffs[t, itm[exercise]]
            
            exercised_prices = X[exercise]
            if len(exercised_prices) > 0:
                boundary_track[t] = np.max(exercised_prices)
            else:
                boundary_track[t] = np.nan
            
        cashflows = cashflows * np.exp(-r * dt)
        if len(itm) > 0:
            cashflows[itm[exercise]] = payoffs[t, itm[exercise]]

    return np.mean(cashflows * np.exp(-r * dt)), boundary_track

## =====================================================================
## 5. EXPERIMENT EXECUTION & COMPARISON SWEEPS
## =====================================================================
print("1/3 Progress: Constructing Tree Ground Truth Benchmark...")
V_truth, tree_times, tree_boundary = merton_tree_benchmark_with_boundary(S0, K, r, sigma, lam, mu_J, sigma_J, T, N_tree_steps=300)

print("2/3 Progress: Simulating Jump-Diffusion Paths...")
S_paths = simulate_merton_paths(S0, r, sigma, lam, mu_J, sigma_J, kappa, T, N_steps, M_paths)

print("3/3 Progress: Running model loops & tracking boundaries...\n")
poly_results = {}
poly_boundaries = {}
lowess_results = {}
lowess_boundaries = {}

# Run standard Polynomial models
poly_degrees = [2, 3, 4]
for degree in poly_degrees:
    price, boundary = price_lsm_american_with_boundary(S_paths, K, r, dt, method='poly', deg=degree)
    poly_results[degree] = price
    poly_boundaries[degree] = boundary

# Run Fixed LOWESS parameter sweep
lowess_spans = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
for alpha in lowess_spans:
    price, boundary = price_lsm_american_with_boundary(S_paths, K, r, dt, method='lowess', alpha=alpha)
    lowess_results[alpha] = price
    lowess_boundaries[alpha] = boundary

## =====================================================================
## 6. FINAL RESULTS REPORTING (ORIGINAL CONSOLE OUTPUT FORMAT)
## =====================================================================
print("=" * 70)
print(f"   EXPERIMENTAL RESULTS & BEST ESTIMATOR EVALUATION")
print("=" * 70)
print(f"Deterministic Tree Benchmark Value ('True Truth'): {V_truth:.5f}\n")

print(f"{'Model Profile':<25}{'Estimated Price':<18}{'Absolute Pricing Bias':<20}")
print("-" * 70)

# Track metrics to isolate the absolute best estimator
best_model_name = ""
min_abs_bias = float('inf')

# Print Polynomial Metrics
for deg, price in poly_results.items():
    bias = price - V_truth
    print(f"Polynomial (Deg {deg})        {price:<18.5f}{bias:<+20.5f}")
    if abs(bias) < min_abs_bias:
        min_abs_bias = abs(bias)
        best_model_name = f"Polynomial (Deg {deg})"

# Print LOWESS Sweep Metrics
for alpha, price in lowess_results.items():
    bias = price - V_truth
    print(f"LOWESS (Fixed α={alpha:.2f})    {price:<18.5f}{bias:<+20.5f}")
    if abs(bias) < min_abs_bias:
        min_abs_bias = abs(bias)
        best_model_name = f"LOWESS (Fixed α={alpha:.2f})"

print("-" * 70)
print(f"Isolated Best Estimator: {best_model_name} (Bias: {min_abs_bias:+.5f})")
print("=" * 70)

## =====================================================================
## 7. MULTI-MODEL BOUNDARY VISUALIZATION
## =====================================================================
mc_time_grid = np.arange(N_steps + 1) * dt

plt.figure(figsize=(15, 9))

# Plot the Ground Truth Control Anchor
plt.plot(tree_times[1:-1], tree_boundary[1:-1], 
         label='GROUND TRUTH (Deterministic Tree)', color='black', lw=4.5, zorder=10)

# Plot Polynomial boundaries
poly_colors = ['crimson', 'darkorange', 'firebrick']
for i, degree in enumerate(poly_degrees):
    plt.plot(mc_time_grid[1:-1], poly_boundaries[degree][1:-1], 
             label=f'Polynomial (Deg {degree})', color=poly_colors[i], lw=2.5, zorder=5)

# Plot LOWESS boundaries using colormap gradient
cmap = plt.get_cmap('viridis')
for i, alpha in enumerate(lowess_spans):
    color_fraction = i / (len(lowess_spans) - 1)
    plt.plot(mc_time_grid[1:-1], lowess_boundaries[alpha][1:-1], 
             label=f'LOWESS (α={alpha:.2f})', color=cmap(color_fraction), linestyle='--', alpha=0.7, lw=1.5, zorder=4)

plt.axhline(K, color='black', linestyle=':', alpha=0.4, label=f'Strike Price (K={K})')

plt.title(f'Exercise Boundary Optimization vs. Theoretical Ground Truth Baseline\nAll Models Swept under Merton Jump-Diffusion ($\lambda$={lam})', 
          fontsize=14, fontweight='bold')
plt.xlabel('Time to Maturity $t$ (Years)', fontsize=11)
plt.ylabel('Critical Exercise Threshold Price $S^*(t)$', fontsize=11)
plt.grid(True, alpha=0.25, linestyle=':')
plt.xlim(0, T)

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., frameon=True)
plt.tight_layout()

print("\nDisplaying comprehensive plot window...")
plt.show()