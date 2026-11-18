"""Linear Algebra & Fitting: examples 36-41.

Run with:  python examples/06_linear_algebra_fitting.py
"""
import numpy as np

# %% 36. Linear regression with lstsq
print("\n=== 36. Linear regression with lstsq ===")
rng = np.random.default_rng(7)
x = np.linspace(0, 10, 50)
y = 3.0 * x + 2.0 + rng.normal(scale=0.5, size=50)
X = np.column_stack([x, np.ones_like(x)])
(slope, intercept), *_ = np.linalg.lstsq(X, y, rcond=None)
print(round(slope, 3), round(intercept, 3))

# %% 37. SVD and low-rank approximation
print("\n=== 37. SVD and low-rank approximation ===")
rng = np.random.default_rng(3)
A = rng.normal(size=(6, 4))
U, s, Vt = np.linalg.svd(A, full_matrices=False)
A1 = (U[:, :1] * s[:1]) @ Vt[:1]
print(np.round(s, 3))
print(np.linalg.matrix_rank(A1))
print(round(np.linalg.norm(A - A1), 3))

# %% 38. eigh for symmetric matrices
print("\n=== 38. eigh for symmetric matrices ===")
S = np.array([[2., 1.], [1., 2.]])
vals, vecs = np.linalg.eigh(S)
print(vals)
print(np.allclose(S @ vecs, vecs * vals))

# %% 39. Vector and matrix norms
print("\n=== 39. Vector and matrix norms ===")
v = np.array([3., 4.])
print(np.linalg.norm(v), np.linalg.norm(v, 1), np.linalg.norm(v, np.inf))
M = np.array([[1., 2.], [3., 4.]])
print(np.linalg.norm(M), np.linalg.norm(M, axis=1))

# %% 40. matrix_power and Fibonacci
print("\n=== 40. matrix_power and Fibonacci ===")
Fib = np.array([[1, 1], [1, 0]])
p = np.linalg.matrix_power(Fib, 10)
print(p)
print(p[0, 1])

# %% 41. Polynomial fit and interpolation
print("\n=== 41. Polynomial fit and interpolation ===")
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 7, 13, 21])      # y = x^2 + x + 1
coef = np.polyfit(x, y, 2)
print(np.round(coef, 3))
print(round(float(np.polyval(coef, 5)), 3))
print(np.interp(2.5, x, y))
