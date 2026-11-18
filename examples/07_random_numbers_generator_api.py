"""Random Numbers (Generator API): examples 42-45.

Run with:  python examples/07_random_numbers_generator_api.py
"""
import numpy as np

# %% 42. default_rng basics
print("\n=== 42. default_rng basics ===")
rng = np.random.default_rng(42)
print(rng.integers(1, 100, size=5))
print(np.round(rng.normal(loc=50, scale=10, size=3), 2))
print(np.round(rng.random(3), 3))

# %% 43. permutation, choice and shuffle
print("\n=== 43. permutation, choice and shuffle ===")
rng = np.random.default_rng(0)
a = np.arange(10)
print(rng.permutation(a))
print(rng.choice(a, size=4, replace=False))
rng.shuffle(a)    # in-place
print(a)

# %% 44. Monte Carlo estimate of pi
print("\n=== 44. Monte Carlo estimate of pi ===")
rng = np.random.default_rng(0)
pts = rng.random((1_000_000, 2))
inside = (pts ** 2).sum(axis=1) <= 1
print(4 * inside.mean())

# %% 45. Bootstrap confidence interval
print("\n=== 45. Bootstrap confidence interval ===")
rng = np.random.default_rng(5)
sample = rng.normal(loc=100, scale=15, size=50)
boot = rng.choice(sample, size=(2000, sample.size), replace=True).mean(axis=1)
print(np.round(np.percentile(boot, [2.5, 97.5]), 2))
