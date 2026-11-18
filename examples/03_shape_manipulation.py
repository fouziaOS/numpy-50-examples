"""Shape Manipulation: examples 15-20.

Run with:  python examples/03_shape_manipulation.py
"""
import numpy as np

# %% 15. reshape with -1 and memory order
print("\n=== 15. reshape with -1 and memory order ===")
a = np.arange(12)
print(a.reshape(3, -1).shape)
print(a.reshape(2, 3, 2).shape)
print(a.reshape(3, 4, order="F")[:, 0])

# %% 16. stack along a new axis
print("\n=== 16. stack along a new axis ===")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.stack([a, b], axis=0))
print(np.stack([a, b], axis=1))

# %% 17. moveaxis and swapaxes
print("\n=== 17. moveaxis and swapaxes ===")
img = np.zeros((4, 5, 3))      # H, W, C
chw = np.moveaxis(img, -1, 0)
print(chw.shape)
print(np.swapaxes(img, 0, 1).shape)

# %% 18. Padding arrays
print("\n=== 18. Padding arrays ===")
print(np.pad(np.ones((2, 2), dtype=int), 1, constant_values=0))
print(np.pad([1, 2, 3], (2, 1), mode="edge"))

# %% 19. roll and flip
print("\n=== 19. roll and flip ===")
a = np.arange(1, 6)
print(np.roll(a, 2))
print(np.flip(a))
m = np.arange(6).reshape(2, 3)
print(np.flip(m, axis=1))

# %% 20. np.block for block matrices
print("\n=== 20. np.block for block matrices ===")
A = np.eye(2, dtype=int)
Z = np.zeros((2, 2), dtype=int)
print(np.block([[A, Z], [Z, 2 * A]]))
