#!/usr/bin/env python3
"""
Optimized LSM American put experiment with polynomial vs LOWESS continuation regression.

Key speedups:
- Fewer default replications and paths for testing
- Removes duplicate scenario B1/A2 duplication
- Uses LOWESS delta parameter for faster interpolation-based smoothing
- Supports selecting a constant LOWESS span once, then reusing it
- Adds progress prints throughout so runtime is visible
- Keeps CV option available, but with smaller default grid/folds

Usage:
    python lsm_lowess_experiment_fast.py
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.nonparametric.smoothers_lowess import lowess

print("[START] Loading script...", flush=True)


@dataclass
class OptionParams:
    S0: float = 40.0
    K: float = 40.0
    r: float = 0.06
    sigma: float = 0.20
    T: float = 1.0
    steps: int = 50


@dataclass
class JumpParams:
    lam: float = 0.05
    muJ: float = -0.10
    sigmaJ: float = 0.20


@dataclass
class ExperimentConfig:
    M: int = 2000
    replications: int = 5
    poly_degree: int = 3
    lowess_fixed_frac: float = 0.25
    lowess_frac_grid: Tuple[float, ...] = (0.12, 0.20, 0.30)
    cv_folds: int = 3
    output_dir: str = "output"
    choose_constant_span_once: bool = True
    constant_span_training_rep: int = 0


class LSMAmericanPutExperiment:
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

    def _cv_select_lowess_frac(self, x, y, frac_grid, folds, seed):
        n = len(x)
        if n < max(30, folds * 5):
            return 0.25
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        fold_indices = np.array_split(idx, folds)
        best_frac, best_mse = None, np.inf
        for frac in frac_grid:
            fold_mse = []
            for fold in fold_indices:
                mask = np.ones(n, dtype=bool)
                mask[fold] = False
                x_train, y_train = x[mask], y[mask]
                x_test, y_test = x[~mask], y[~mask]
                y_pred = self._lowess_predict_interp(x_train, y_train, x_test, frac=frac)
                fold_mse.append(np.mean((y_test - y_pred) ** 2))
            avg_mse = float(np.mean(fold_mse))
            if avg_mse < best_mse:
                best_mse, best_frac = avg_mse, frac
        return float(best_frac)

    def choose_global_lowess_frac(self, S, frac_grid, folds):
        payoff = np.maximum(self.option.K - S, 0.0)
        candidate_steps = [10, 20, 30, 40]
        scores = []
        print("[LOWESS] Choosing one constant span using representative steps...", flush=True)
        for t in candidate_steps:
            itm = payoff[:, t] > 0.0
            if np.sum(itm) < 20:
                continue
            x = S[itm, t]
            y = payoff[itm, t + 1] * self.discount
            best = self._cv_select_lowess_frac(x, y, frac_grid, folds, seed=5000 + t)
            scores.append(best)
            print(f"  step={t} best_frac={best:.3f} itm={np.sum(itm)}", flush=True)
        if not scores:
            return 0.25
        global_frac = float(np.median(scores))
        print(f"[LOWESS] Selected constant frac={global_frac:.3f}", flush=True)
        return global_frac

    def price_american_put_lsm(self, S, regression='polynomial', poly_degree=3,
                               lowess_frac=0.25, span_method='fixed',
                               frac_grid=(0.12, 0.20, 0.30), cv_folds=3,
                               progress_label=''):
        M, ncols = S.shape
        steps = ncols - 1
        payoff = np.maximum(self.option.K - S, 0.0)
        cashflow = payoff[:, -1].copy()
        boundary_rows = []
        frac_rows = []

        if progress_label:
            print(f"    [LSM] {progress_label} backward induction starting...", flush=True)

        for t in range(steps - 1, 0, -1):
            if t % 10 == 0 and progress_label:
                print(f"      step {t}/{steps}", flush=True)

            itm = payoff[:, t] > 0.0
            cashflow *= self.discount

            if np.sum(itm) < 20:
                boundary_rows.append({'step': t, 'time': self.times[t], 'boundary_S': np.nan})
                continue

            x = S[itm, t]
            y = cashflow[itm]

            if regression == 'polynomial':
                continuation = self._poly_predict(x, y, x, degree=poly_degree)
                frac_used = np.nan
            else:
                if span_method == 'fixed':
                    frac_used = lowess_frac
                elif span_method == 'cv':
                    frac_used = self._cv_select_lowess_frac(x, y, frac_grid, cv_folds, seed=1000 + t)
                else:
                    raise ValueError("span_method must be 'fixed' or 'cv'")
                continuation = self._lowess_predict_interp(x, y, x, frac=frac_used)
                frac_rows.append({'step': t, 'time': self.times[t], 'frac': frac_used, 'itm_count': int(len(x))})

            intrinsic = payoff[itm, t]
            exercise_now = intrinsic > continuation
            exercise_idx = np.where(itm)[0][exercise_now]
            cashflow[exercise_idx] = payoff[exercise_idx, t]

            xs = np.sort(x)
            intrinsic_sorted = np.maximum(self.option.K - xs, 0.0)
            if regression == 'polynomial':
                cont_sorted = self._poly_predict(x, y, xs, degree=poly_degree)
            else:
                cont_sorted = self._lowess_predict_interp(x, y, xs, frac=frac_used)
            diff = intrinsic_sorted - cont_sorted
            cross_idx = np.where(diff[:-1] * diff[1:] <= 0)[0]
            boundary_S = float(xs[cross_idx[0]]) if len(cross_idx) > 0 else np.nan
            boundary_rows.append({'step': t, 'time': self.times[t], 'boundary_S': boundary_S})

        price_paths = cashflow * self.discount
        price = float(np.mean(price_paths))
        se = float(np.std(price_paths, ddof=1) / math.sqrt(M))
        boundary_df = pd.DataFrame(boundary_rows).sort_values('step')
        frac_df = pd.DataFrame(frac_rows)
        return price, se, boundary_df, frac_df


def run_full_experiment(option, jump_default, config):
    os.makedirs(config.output_dir, exist_ok=True)
    exp = LSMAmericanPutExperiment(option)

    scenarios = [
        {'scenario': 'A1', 'model': 'GBM', 'lam': 0.00, 'regression': 'polynomial', 'span_method': 'fixed'},
        {'scenario': 'A2', 'model': 'Jump-diffusion', 'lam': 0.05, 'regression': 'polynomial', 'span_method': 'fixed'},
        {'scenario': 'B2_fixed', 'model': 'Jump-diffusion', 'lam': 0.05, 'regression': 'lowess', 'span_method': 'fixed'},
        {'scenario': 'B2_cv', 'model': 'Jump-diffusion', 'lam': 0.05, 'regression': 'lowess', 'span_method': 'cv'},
    ]

    print("[RUN] Starting experiment", flush=True)
    print(f"[RUN] Paths={config.M}, replications={config.replications}, steps={option.steps}", flush=True)
    print(f"[RUN] LOWESS fixed frac={config.lowess_fixed_frac}, CV grid={config.lowess_frac_grid}, folds={config.cv_folds}", flush=True)

    results_rows, boundary_frames, frac_frames = [], [], []
    eur_gbm = exp.bs_european_put()
    eur_jump_cache: Dict[float, float] = {}
    constant_frac = config.lowess_fixed_frac

    if config.choose_constant_span_once:
        print("[SETUP] Precomputing one constant LOWESS span from a pilot jump-diffusion run...", flush=True)
        pilot_seed = 2026000 + config.constant_span_training_rep
        pilot_jump = JumpParams(lam=0.05, muJ=jump_default.muJ, sigmaJ=jump_default.sigmaJ)
        pilot_paths = exp.simulate_paths_jump(M=config.M, jump=pilot_jump, seed=pilot_seed)
        constant_frac = exp.choose_global_lowess_frac(pilot_paths, config.lowess_frac_grid, config.cv_folds)

    for rep in range(config.replications):
        rep_start = time.time()
        seed = 2026000 + rep
        print(f"\n[REP] {rep + 1}/{config.replications} | seed={seed}", flush=True)

        for sc in scenarios:
            scen_start = time.time()
            lam = sc['lam']
            regression = sc['regression']
            span_method = sc['span_method']
            model = sc['model']
            print(f"  [SCENARIO] {sc['scenario']} | {model} | {regression} | {span_method}", flush=True)

            jump = JumpParams(lam=lam, muJ=jump_default.muJ, sigmaJ=jump_default.sigmaJ)

            if model == 'GBM':
                print("    simulating GBM paths...", flush=True)
                S = exp.simulate_paths_gbm(M=config.M, seed=seed)
                eur = eur_gbm
            else:
                print(f"    simulating jump-diffusion paths (lambda={lam})...", flush=True)
                S = exp.simulate_paths_jump(M=config.M, jump=jump, seed=seed)
                if lam not in eur_jump_cache:
                    eur_jump_cache[lam] = exp.merton_european_put_series(jump=jump)
                eur = eur_jump_cache[lam]

            frac_to_use = constant_frac if (regression == 'lowess' and span_method == 'fixed' and config.choose_constant_span_once) else config.lowess_fixed_frac
            price, mc_se, boundary_df, frac_df = exp.price_american_put_lsm(
                S=S,
                regression=regression,
                poly_degree=config.poly_degree,
                lowess_frac=frac_to_use,
                span_method=span_method,
                frac_grid=config.lowess_frac_grid,
                cv_folds=config.cv_folds,
                progress_label=f"rep {rep + 1} {sc['scenario']}"
            )

            eep = price - eur
            elapsed = time.time() - scen_start
            print(f"    done in {elapsed:.2f}s | American={price:.4f} | European={eur:.4f} | EEP={eep:.4f} | MC SE={mc_se:.5f}", flush=True)

            results_rows.append({
                'replication': rep + 1,
                'scenario': sc['scenario'],
                'model': model,
                'lambda': lam,
                'regression': regression,
                'span_method': span_method,
                'fixed_lowess_frac': frac_to_use if regression == 'lowess' and span_method == 'fixed' else np.nan,
                'american_price': price,
                'mc_se_within_run': mc_se,
                'european_price': eur,
                'early_exercise_premium': eep,
            })

            if not boundary_df.empty:
                tmp = boundary_df.copy()
                tmp['replication'] = rep + 1
                tmp['scenario'] = sc['scenario']
                tmp['model'] = model
                tmp['regression'] = regression
                tmp['span_method'] = span_method
                boundary_frames.append(tmp)

            if not frac_df.empty:
                tmpf = frac_df.copy()
                tmpf['replication'] = rep + 1
                tmpf['scenario'] = sc['scenario']
                frac_frames.append(tmpf)

        print(f"[REP] completed in {time.time() - rep_start:.2f}s", flush=True)

    print("\n[SAVE] Writing outputs...", flush=True)
    results = pd.DataFrame(results_rows)
    boundaries_all = pd.concat(boundary_frames, ignore_index=True) if boundary_frames else pd.DataFrame()
    fracs_all = pd.concat(frac_frames, ignore_index=True) if frac_frames else pd.DataFrame()

    summary = results.groupby(['scenario', 'model', 'lambda', 'regression', 'span_method'], as_index=False).agg(
        american_mean=('american_price', 'mean'),
        american_sd_across_reps=('american_price', 'std'),
        avg_mc_se=('mc_se_within_run', 'mean'),
        european_mean=('european_price', 'mean'),
        eep_mean=('early_exercise_premium', 'mean')
    )
    summary['ci95_halfwidth'] = 1.96 * summary['american_sd_across_reps'] / np.sqrt(config.replications)

    boundary_summary = pd.DataFrame()
    if not boundaries_all.empty:
        boundary_summary = boundaries_all.groupby(['scenario', 'model', 'regression', 'span_method', 'step', 'time'], as_index=False).agg(
            boundary_mean=('boundary_S', 'mean'),
            boundary_sd=('boundary_S', 'std')
        )

    frac_summary = pd.DataFrame()
    if not fracs_all.empty:
        frac_summary = fracs_all.groupby(['scenario', 'step', 'time'], as_index=False).agg(
            frac_mean=('frac', 'mean'),
            frac_sd=('frac', 'std'),
            itm_avg=('itm_count', 'mean')
        )

    results.to_csv(os.path.join(config.output_dir, 'lsm_full_results_fast.csv'), index=False)
    summary.to_csv(os.path.join(config.output_dir, 'lsm_summary_fast.csv'), index=False)
    if not boundaries_all.empty:
        boundaries_all.to_csv(os.path.join(config.output_dir, 'lsm_boundaries_all_fast.csv'), index=False)
        boundary_summary.to_csv(os.path.join(config.output_dir, 'lsm_boundary_summary_fast.csv'), index=False)
    if not fracs_all.empty:
        fracs_all.to_csv(os.path.join(config.output_dir, 'lsm_lowess_fracs_all_fast.csv'), index=False)
        frac_summary.to_csv(os.path.join(config.output_dir, 'lsm_lowess_frac_summary_fast.csv'), index=False)

    with open(os.path.join(config.output_dir, 'lsm_experiment_report_fast.txt'), 'w', encoding='utf-8') as f:
        f.write('Optimized LSM LOWESS experiment\n')
        f.write('================================\n\n')
        f.write(f'Option parameters: S0={option.S0}, K={option.K}, r={option.r}, sigma={option.sigma}, T={option.T}, steps={option.steps}\n')
        f.write(f'Replications: {config.replications}\n')
        f.write(f'Paths per replication: {config.M}\n')
        f.write(f'Polynomial degree: {config.poly_degree}\n')
        f.write(f'LOWESS fixed frac used: {constant_frac if config.choose_constant_span_once else config.lowess_fixed_frac}\n')
        f.write(f'LOWESS CV grid: {config.lowess_frac_grid}\n')
        f.write(f'LOWESS CV folds: {config.cv_folds}\n\n')
        f.write(summary.to_string(index=False))

    print("[DONE] Experiment complete.", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"[DONE] Files saved in: {config.output_dir}", flush=True)


def main():
    print("[MAIN] Entered main()", flush=True)
    option = OptionParams(S0=40.0, K=40.0, r=0.06, sigma=0.20, T=1.0, steps=50)
    jump_default = JumpParams(lam=0.05, muJ=-0.10, sigmaJ=0.20)
    config = ExperimentConfig(
        M=2000,
        replications=5,
        poly_degree=3,
        lowess_fixed_frac=0.25,
        lowess_frac_grid=(0.12, 0.20, 0.30),
        cv_folds=3,
        output_dir='output',
        choose_constant_span_once=True,
        constant_span_training_rep=0,
    )
    run_full_experiment(option, jump_default, config)


if __name__ == '__main__':
    main()