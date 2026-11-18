"""Statistics & Data Analysis: examples 28-35.

Run with:  python examples/05_statistics_data_analysis.py
"""
import numpy as np

# %% 28. Percentiles and IQR outlier detection
print("\n=== 28. Percentiles and IQR outlier detection ===")
data = np.array([12, 15, 11, 18, 21, 14, 90])
print(np.median(data), data.mean())
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
print(data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)])

# %% 29. Correlation matrix
print("\n=== 29. Correlation matrix ===")
rng = np.random.default_rng(0)
x = rng.normal(size=100)
y = 2 * x + rng.normal(scale=0.5, size=100)
print(np.round(np.corrcoef(x, y), 3))

# %% 30. bincount and histogram
print("\n=== 30. bincount and histogram ===")
rng = np.random.default_rng(1)
data = rng.integers(0, 5, size=20)
print(data)
print(np.bincount(data))
counts, edges = np.histogram(rng.normal(size=1000), bins=5)
print(counts)
print(np.round(edges, 2))

# %% 31. digitize for binning
print("\n=== 31. digitize for binning ===")
ages = np.array([5, 17, 25, 42, 67, 80])
bins = np.array([18, 40, 65])
labels = np.array(["child", "young", "adult", "senior"])
print(labels[np.digitize(ages, bins)])

# %% 32. Weighted average
print("\n=== 32. Weighted average ===")
grades = np.array([80, 90, 70])
credits = np.array([3, 4, 2])
print(np.average(grades, weights=credits))

# %% 33. diff and gradient
print("\n=== 33. diff and gradient ===")
y = np.array([1, 4, 9, 16, 25])
print(np.diff(y))
print(np.gradient(y))

# %% 34. Moving average with convolve
print("\n=== 34. Moving average with convolve ===")
x = np.arange(1, 11, dtype=float)
w = 3
print(np.convolve(x, np.ones(w) / w, mode="valid"))

# %% 35. Running maximum
print("\n=== 35. Running maximum ===")
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(np.maximum.accumulate(a))
