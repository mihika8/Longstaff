#!/usr/bin/env python3
"""
Benchmark script for American put pricing experiments.

What it does:
1. Computes high-precision benchmark American prices using Longstaff-Schwartz
   with many more paths and time steps than the fast experiment.
2. Uses both polynomial and LOWESS regression under the same scenario set.
3. Reads the fast experiment results (if available) and compares them to the
   benchmark values.
4. Writes CSV files with benchmark prices and error metrics.

Run:
    python lsm_benchmark_compare.py
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.nonparametric.smoothers_lowess import lowess

print("[START] Benchmark script starting...", flush=True)


@dataclass
class OptionParams:
    S0: float = 40.0
    K: float = 40.0
    r: float = 0.06
    sigma: float = 0.20
    T: float = 1.0
    steps: int = 100


@dataclass
class JumpParams:
    lam: float = 0.05
    muJ: float = -0.10
    sigmaJ: float = 0.20


@dataclass
class BenchmarkConfig:
    M: int = 50000
    poly_degree: int = 3
    lowess_frac: float = 0.20
    output_dir: str = "output"
    read_fast_results: bool = True
    fast_results_file: str = "output/lsm_full_results_fast.csv"
    replications_to_compare: int = 1


class BenchmarkLSM:
    def __init__(self, option: OptionParams):
        self.option = option
        self.dt = option.T / option.steps
        self.discount = math.exp(-option.r * self.dt)
        self.times = np.linspace(0.0, option.T, option.steps + 1)

    def bs_european_put(self) -> float:
        S0, K, r, sigma, T = self.option.S0, self.option.K, self.option.r, self.option.sigma, self.option.T
        d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    def merton_european_put_series(self, jump: JumpParams, n_terms: int = 50) -> float:
        S0, K, r, sigma, T = self.option.S0, self.option.K, self.option.r, self.option.sigma, self.option.T
        lam, muJ, sigmaJ = jump.lam, jump.muJ, jump.sigmaJ
        kappa = math.exp(muJ + 0.5 * sigmaJ**2) - 1.0
        lamT = lam * T
        total = 0.0
        for n in range(n_terms):
            w = math.exp(-lamT) * lamT**n / math.factorial(n)
            sigma_n = math.sqrt(sigma**2 + (n * sigmaJ**2) / T)
            r_n = r - lam * kappa + n * (muJ + 0.5 * sigmaJ**2) / T
            d1 = (math.log(S0 / K) + (r_n + 0.5 * sigma_n**2) * T) / (sigma_n * math.sqrt(T))
            d2 = d1 - sigma_n * math.sqrt(T)
            put_n = K * math.exp(-r_n * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
            total += w * put_n
        return total

    def simulate_paths_gbm(self, M: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        dt = self.dt
        steps = self.option.steps
        S = np.empty((M, steps + 1), dtype=float)
        S[:, 0] = self.option.S0
        drift = (self.option.r - 0.5 * self.option.sigma**2) * dt
        vol = self.option.sigma * math.sqrt(dt)
        Z = rng.standard_normal((M, steps))
        for t in range(steps):
            S[:, t + 1] = S[:, t] * np.exp(drift + vol * Z[:, t])
        return S

    def simulate_paths_jump(self, M: int, jump: JumpParams, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        dt = self.dt
        steps = self.option.steps
        lam, muJ, sigmaJ = jump.lam, jump.muJ, jump.sigmaJ
        S = np.empty((M, steps + 1), dtype=float)
        S[:, 0] = self.option.S0
        kappa = math.exp(muJ + 0.5 * sigmaJ**2) - 1.0
        drift = (self.option.r - lam * kappa - 0.5 * self.option.sigma**2) * dt
        vol = self.option.sigma * math.sqrt(dt)
        Z = rng.standard_normal((M, steps))
        N = rng.poisson(lam * dt, size=(M, steps))
        for t in range(steps):
            Nt = N[:, t]
            jump_sum = np.zeros(M)
            positive = Nt > 0
            if np.any(positive):
                jump_sum[positive] = Nt[positive] * muJ + np.sqrt(Nt[positive]) * sigmaJ * rng.standard_normal(np.sum(positive))
            S[:, t + 1] = S[:, t] * np.exp(drift + vol * Z[:, t] + jump_sum)
        return S

    @staticmethod
    def _poly_predict(x_train, y_train, x_pred, degree=3):
        max_degree = max(1, min(degree, len(np.unique(x_train)) - 1))
        coeffs = np.polyfit(x_train, y_train, max_degree)
        return np.polyval(coeffs, x_pred)

    @staticmethod
    def _lowess_predict_interp(x_train, y_train, x_pred, frac):
        x_range = float(np.max(x_train) - np.min(x_train)) if len(x_train) else 0.0
        delta = 0.01 * x_range if x_range > 0 else 0.0
        fitted = lowess(endog=y_train, exog=x_train, frac=frac, it=0, delta=delta, return_sorted=True)
        xf, yf = fitted[:, 0], fitted[:, 1]
        ux, idx = np.unique(xf, return_index=True)
        uy = yf[idx]
        return np.interp(x_pred, ux, uy, left=uy[0], right=uy[-1])

    def price_american_put_lsm(self, S, regression='polynomial', poly_degree=3, lowess_frac=0.20, progress_label=''):
        M, ncols = S.shape
        steps = ncols - 1
        payoff = np.maximum(self.option.K - S, 0.0)
        cashflow = payoff[:, -1].copy()

        if progress_label:
            print(f"    [LSM] {progress_label} backward induction...", flush=True)

        for t in range(steps - 1, 0, -1):
            if t % 20 == 0 and progress_label:
                print(f"      step {t}/{steps}", flush=True)

            itm = payoff[:, t] > 0.0
            cashflow *= self.discount
            if np.sum(itm) < 20:
                continue
            x = S[itm, t]
            y = cashflow[itm]
            if regression == 'polynomial':
                continuation = self._poly_predict(x, y, x, degree=poly_degree)
            else:
                continuation = self._lowess_predict_interp(x, y, x, frac=lowess_frac)
            intrinsic = payoff[itm, t]
            exercise_now = intrinsic > continuation
            exercise_idx = np.where(itm)[0][exercise_now]
            cashflow[exercise_idx] = payoff[exercise_idx, t]

        price_paths = cashflow * self.discount
        price = float(np.mean(price_paths))
        se = float(np.std(price_paths, ddof=1) / math.sqrt(M))
        early_exercise_premium = np.nan
        return price, se, early_exercise_premium


def main():
    option = OptionParams(S0=40.0, K=40.0, r=0.06, sigma=0.20, T=1.0, steps=100)
    jump = JumpParams(lam=0.05, muJ=-0.10, sigmaJ=0.20)
    config = BenchmarkConfig(M=50000, poly_degree=3, lowess_frac=0.20, output_dir='output')
    os.makedirs(config.output_dir, exist_ok=True)

    bench = BenchmarkLSM(option)
    benchmark_rows = []

    scenarios = [
        ('A1', 'GBM', 0.0, 'polynomial'),
        ('A2_benchmark_poly', 'Jump-diffusion', 0.05, 'polynomial'),
        ('B2_benchmark_lowess', 'Jump-diffusion', 0.05, 'lowess'),
    ]

    print(f"[RUN] Benchmark paths={config.M}, steps={option.steps}", flush=True)

    for i, (scenario, model, lam, regression) in enumerate(scenarios, start=1):
        print(f"\n[SCENARIO] {i}/{len(scenarios)} {scenario} | {model} | {regression}", flush=True)
        t0 = time.time()

        if model == 'GBM':
            print("  simulating benchmark GBM paths...", flush=True)
            S = bench.simulate_paths_gbm(config.M, seed=909000 + i)
            eur = bench.bs_european_put()
        else:
            print(f"  simulating benchmark jump-diffusion paths (lambda={lam})...", flush=True)
            S = bench.simulate_paths_jump(config.M, jump=jump, seed=909000 + i)
            eur = bench.merton_european_put_series(jump)

        price, se, _ = bench.price_american_put_lsm(
            S=S,
            regression='polynomial' if regression == 'polynomial' else 'lowess',
            poly_degree=config.poly_degree,
            lowess_frac=config.lowess_frac,
            progress_label=scenario
        )

        eep = price - eur
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.2f}s | benchmark American={price:.6f} | benchmark SE={se:.6f} | European={eur:.6f} | EEP={eep:.6f}", flush=True)

        benchmark_rows.append({
            'benchmark_scenario': scenario,
            'model': model,
            'lambda': lam,
            'regression': regression,
            'benchmark_american_price': price,
            'benchmark_mc_se': se,
            'benchmark_european_price': eur,
            'benchmark_early_exercise_premium': eep,
            'benchmark_paths': config.M,
            'benchmark_steps': option.steps,
            'benchmark_lowess_frac': config.lowess_frac if regression == 'lowess' else np.nan,
        })

    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_file = os.path.join(config.output_dir, 'benchmark_prices.csv')
    benchmark_df.to_csv(benchmark_file, index=False)
    print(f"\n[SAVE] Wrote benchmark prices to {benchmark_file}", flush=True)

    if config.read_fast_results and os.path.exists(config.fast_results_file):
        print(f"[COMPARE] Reading fast results from {config.fast_results_file}", flush=True)
        fast = pd.read_csv(config.fast_results_file)

        mapping = {
            'A1': 'A1',
            'A2': 'A2_benchmark_poly',
            'B2_fixed': 'B2_benchmark_lowess',
            'B2_cv': 'B2_benchmark_lowess',
        }
        fast['benchmark_scenario'] = fast['scenario'].map(mapping)
        merged = fast.merge(benchmark_df, on='benchmark_scenario', how='left', suffixes=('', '_bench'))
        merged['abs_error_vs_benchmark'] = (merged['american_price'] - merged['benchmark_american_price']).abs()
        merged['signed_error_vs_benchmark'] = merged['american_price'] - merged['benchmark_american_price']
        merged['rel_error_vs_benchmark'] = merged['signed_error_vs_benchmark'] / merged['benchmark_american_price']

        compare_file = os.path.join(config.output_dir, 'benchmark_comparison_full.csv')
        merged.to_csv(compare_file, index=False)
        print(f"[SAVE] Wrote full benchmark comparison to {compare_file}", flush=True)

        summary = merged.groupby(['scenario', 'model', 'regression', 'span_method'], as_index=False).agg(
            american_mean=('american_price', 'mean'),
            benchmark_american_price=('benchmark_american_price', 'mean'),
            mean_abs_error=('abs_error_vs_benchmark', 'mean'),
            mean_signed_error=('signed_error_vs_benchmark', 'mean'),
            mean_rel_error=('rel_error_vs_benchmark', 'mean')
        )
        summary['rank_by_abs_error'] = summary['mean_abs_error'].rank(method='dense')
        summary = summary.sort_values('mean_abs_error')

        summary_file = os.path.join(config.output_dir, 'benchmark_comparison_summary.csv')
        summary.to_csv(summary_file, index=False)
        print(f"[SAVE] Wrote benchmark comparison summary to {summary_file}", flush=True)
        print("\n[RESULT] Accuracy ranking (lower mean_abs_error is better):", flush=True)
        print(summary.to_string(index=False), flush=True)
    else:
        print("[SKIP] Fast experiment results file not found; benchmark prices saved only.", flush=True)

    report_file = os.path.join(config.output_dir, 'benchmark_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('Benchmark pricing report\n')
        f.write('========================\n\n')
        f.write(benchmark_df.to_string(index=False))
        f.write('\n\nInterpretation:\n')
        f.write('- These benchmark prices are higher-precision references built with more paths and more exercise dates than the fast experiment.\n')
        f.write('- Use benchmark_comparison_summary.csv to identify which fast method is closest to the benchmark.\n')
        f.write('- Lower mean_abs_error indicates the more accurate pricing method relative to the benchmark.\n')

    print(f"[DONE] Benchmark report saved to {report_file}", flush=True)
    print("[DONE] Benchmark script complete.", flush=True)


if __name__ == '__main__':
    main()