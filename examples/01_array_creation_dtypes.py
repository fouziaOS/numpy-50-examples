"""Array Creation & Dtypes: examples 1-6.

Run with:  python examples/01_array_creation_dtypes.py
"""
import numpy as np

# %% 1. Explicit dtypes and memory footprint
print("\n=== 1. Explicit dtypes and memory footprint ===")
a = np.array([1, 2, 3], dtype=np.float32)
print(a.dtype, a.itemsize, a.nbytes)
b = np.array([1, 2, 3], dtype=np.float64)
print(b.dtype, b.itemsize, b.nbytes)

# %% 2. arange vs linspace vs logspace
print("\n=== 2. arange vs linspace vs logspace ===")
print(np.arange(0, 1, 0.25))
print(np.linspace(0, 1, 5))   # includes endpoint
print(np.logspace(0, 3, 4))   # 10**0 ... 10**3

# %% 3. Build arrays from a function
print("\n=== 3. Build arrays from a function ===")
grid = np.fromfunction(lambda i, j: i * 3 + j, (3, 4), dtype=int)
print(grid)

# %% 4. Structured arrays (mini tables)
print("\n=== 4. Structured arrays (mini tables) ===")
dt = np.dtype([("name", "U10"), ("age", "i4"), ("score", "f8")])
people = np.array([("Asha", 25, 88.5), ("Rahim", 31, 72.0), ("Mitu", 28, 91.2)], dtype=dt)
print(people["name"])
print(people[people["age"] > 26])
print(np.sort(people, order="score")[::-1]["name"])

# %% 5. diag, triu and tri
print("\n=== 5. diag, triu and tri ===")
print(np.diag([1, 2, 3]))
print(np.triu(np.arange(1, 10).reshape(3, 3)))
print(np.tri(3, dtype=int))

# %% 6. Integer overflow and safe widening
print("\n=== 6. Integer overflow and safe widening ===")
x = np.array([200], dtype=np.uint8)
print(x + 100)                     # wraps around silently
print(np.iinfo(np.uint8).max)
print(x.astype(np.int16) + 100)    # widen first to be safe
