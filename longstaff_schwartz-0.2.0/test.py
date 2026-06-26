print("Starting test...")
from longstaff_schwartz.algorithm import longstaff_schwartz as lsmc_algorithm
from longstaff_schwartz.stochastic_process import GeometricBrownianMotion
from longstaff_schwartz.utils import constant_rate_discount
import numpy as np
print("✓ All imports successful!")

# Quick test
gbm = GeometricBrownianMotion(mu=0.05, sigma=0.25)
times = np.linspace(0, 1, 100)
rng = np.random.RandomState(42)
paths = gbm.simulate(times, 10, rng)
print(f"✓ Simulated {paths.shape[1]} paths with {paths.shape[0]} timesteps")
print(f"✓ Initial price: {paths[0,0]:.4f}")
print("\n✅ Everything works! Ready to run full script.")
