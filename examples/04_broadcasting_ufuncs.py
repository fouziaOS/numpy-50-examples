"""Broadcasting & Ufuncs: examples 21-27.

Run with:  python examples/04_broadcasting_ufuncs.py
"""
import numpy as np

# %% 21. Pairwise distance matrix without loops
print("\n=== 21. Pairwise distance matrix without loops ===")
pts = np.array([[0, 0], [3, 4], [6, 8]])
diff = pts[:, None, :] - pts[None, :, :]
dist = np.sqrt((diff ** 2).sum(axis=-1))
print(dist)

# %% 22. Column-wise z-score with keepdims
print("\n=== 22. Column-wise z-score with keepdims ===")
X = np.array([[1., 200.], [2., 400.], [3., 600.]])
Z = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
print(np.round(Z, 3))

# %% 23. ufunc reduce, accumulate and outer
print("\n=== 23. ufunc reduce, accumulate and outer ===")
a = np.arange(1, 5)
print(np.add.reduce(a), np.multiply.reduce(a))
print(np.add.accumulate(a))
print(np.multiply.outer(np.arange(1, 4), np.arange(1, 4)))

# %% 24. np.add.at for repeated indices
print("\n=== 24. np.add.at for repeated indices ===")
idx = np.array([0, 0, 2, 4, 4, 4])
counts = np.zeros(5, dtype=int)
np.add.at(counts, idx, 1)
print(counts)
counts2 = np.zeros(5, dtype=int)
counts2[idx] += 1
print(counts2)    # buffered: repeats are lost

# %% 25. np.vectorize (convenience, not speed)
print("\n=== 25. np.vectorize (convenience, not speed) ===")
def collatz_step(n):
    return n // 2 if n % 2 == 0 else 3 * n + 1

step = np.vectorize(collatz_step)
print(step(np.arange(1, 8)))

# %% 26. einsum for tensor operations
print("\n=== 26. einsum for tensor operations ===")
A = np.arange(6).reshape(2, 3)
B = np.arange(6).reshape(3, 2)
print(np.einsum("ij,jk->ik", A, B))
print(np.einsum("ij->ji", A).shape)
print(np.einsum("ii", np.eye(3)))
batch = np.ones((4, 2, 3))
print(np.einsum("bij,bjk->bik", batch, np.ones((4, 3, 5))).shape)

# %% 27. clip with in-place output
print("\n=== 27. clip with in-place output ===")
a = np.array([-5, 0, 5, 10, 15])
print(np.clip(a, 0, 10))
np.clip(a, 0, 10, out=a)
print(a)
