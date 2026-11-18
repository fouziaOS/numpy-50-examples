"""Practical Patterns: examples 46-50.

Run with:  python examples/08_practical_patterns.py
"""
import numpy as np

# %% 46. sliding_window_view
print("\n=== 46. sliding_window_view ===")
from numpy.lib.stride_tricks import sliding_window_view

x = np.arange(1, 8)
win = sliding_window_view(x, 3)
print(win)
print(win.mean(axis=1))

# %% 47. Masked arrays
print("\n=== 47. Masked arrays ===")
data = np.ma.masked_values([1.0, -999.0, 3.0, -999.0, 5.0], -999.0)
print(data)
print(data.mean())
print(data.filled(0))

# %% 48. NaN handling and mean imputation
print("\n=== 48. NaN handling and mean imputation ===")
X = np.array([[1., 2.], [np.nan, 4.], [5., np.nan]])
col_mean = np.nanmean(X, axis=0)
filled = np.where(np.isnan(X), col_mean, X)
print(col_mean)
print(filled)

# %% 49. Stable softmax and one-hot encoding
print("\n=== 49. Stable softmax and one-hot encoding ===")
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

print(np.round(softmax(np.array([1000., 1001., 1002.])), 3))
labels = np.array([0, 2, 1, 2])
print(np.eye(3, dtype=int)[labels])

# %% 50. Save and load with .npz
print("\n=== 50. Save and load with .npz ===")
import os, tempfile

path = os.path.join(tempfile.gettempdir(), "demo.npz")
np.savez_compressed(path, a=np.arange(5), b=np.eye(2))
loaded = np.load(path)
print(loaded.files)
print(loaded["a"], loaded["b"].shape)
