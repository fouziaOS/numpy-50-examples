"""Indexing & Selection: examples 7-14.

Run with:  python examples/02_indexing_selection.py
"""
import numpy as np

# %% 7. Views vs copies
print("\n=== 7. Views vs copies ===")
a = np.arange(6)
view = a[1:4]
view[0] = 99
print(a)
print(np.shares_memory(a, view))
copy = a[[1, 2, 3]]            # fancy indexing -> copy
print(np.shares_memory(a, copy))

# %% 8. Fancy indexing with index arrays
print("\n=== 8. Fancy indexing with index arrays ===")
m = np.arange(12).reshape(3, 4)
print(m)
print(m[[0, 2], [1, 3]])       # pairs: (0,1) and (2,3)
print(m[[0, 2]][:, [1, 3]])    # rows 0,2 then columns 1,3

# %% 9. np.ix_ for sub-grids
print("\n=== 9. np.ix_ for sub-grids ===")
m = np.arange(12).reshape(3, 4)
print(m[np.ix_([0, 2], [1, 3])])

# %% 10. Boolean masks with multiple conditions
print("\n=== 10. Boolean masks with multiple conditions ===")
a = np.arange(20)
mask = (a % 2 == 0) & (a > 10)
print(mask.sum())
print(a[mask])
print(a[~mask][:5])

# %% 11. argwhere and nonzero
print("\n=== 11. argwhere and nonzero ===")
m = np.array([[1, 0, 3], [0, 5, 0]])
print(np.argwhere(m > 0))
rows, cols = np.nonzero(m)
print(rows, cols)

# %% 12. take_along_axis with argsort
print("\n=== 12. take_along_axis with argsort ===")
m = np.array([[3, 1, 2], [9, 7, 8]])
idx = np.argsort(m, axis=1)
print(idx)
print(np.take_along_axis(m, idx, axis=1))

# %% 13. newaxis and Ellipsis
print("\n=== 13. newaxis and Ellipsis ===")
x = np.arange(3)
print(x[:, np.newaxis].shape, x[None, :].shape)
t = np.zeros((2, 3, 4))
print(t[..., 0].shape)
print(t[0, ...].shape)

# %% 14. np.select for multi-branch conditions
print("\n=== 14. np.select for multi-branch conditions ===")
scores = np.array([95, 72, 58, 81, 66])
grades = np.select([scores >= 80, scores >= 60], ["A", "B"], default="C")
print(grades)
