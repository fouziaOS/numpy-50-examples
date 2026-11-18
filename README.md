# NumPy in 50 Examples

A hands-on collection of 50 runnable NumPy examples, from dtype tricks and fancy indexing to broadcasting, linear algebra, the modern random `Generator` API and real-world patterns. Every snippet comes with its actual output.

> Outputs were generated with **NumPy 2.5.4** and Python 3.12.3. Random results use fixed seeds, but exact numbers can differ slightly between NumPy versions.

## Quick start

```bash
git clone https://github.com/fouziaOS/numpy-50-examples.git
cd numpy-50-examples
pip install -r requirements.txt
python examples/01_array_creation_dtypes.py
```

Prefer notebooks? Open `numpy_50_examples.ipynb` in Jupyter, VS Code or Colab.

## Contents

- [Array Creation & Dtypes](#array-creation--dtypes) (examples 1-6)
- [Indexing & Selection](#indexing--selection) (examples 7-14)
- [Shape Manipulation](#shape-manipulation) (examples 15-20)
- [Broadcasting & Ufuncs](#broadcasting--ufuncs) (examples 21-27)
- [Statistics & Data Analysis](#statistics--data-analysis) (examples 28-35)
- [Linear Algebra & Fitting](#linear-algebra--fitting) (examples 36-41)
- [Random Numbers (Generator API)](#random-numbers-generator-api) (examples 42-45)
- [Practical Patterns](#practical-patterns) (examples 46-50)

## Array Creation & Dtypes

Source file: [`examples/01_array_creation_dtypes.py`](examples/01_array_creation_dtypes.py)

### 1. Explicit dtypes and memory footprint

Choosing the right dtype halves (or doubles) memory. Inspect it with `itemsize` and `nbytes`.

```python
a = np.array([1, 2, 3], dtype=np.float32)
print(a.dtype, a.itemsize, a.nbytes)
b = np.array([1, 2, 3], dtype=np.float64)
print(b.dtype, b.itemsize, b.nbytes)
```

Output:

```text
float32 4 12
float64 8 24
```

### 2. arange vs linspace vs logspace

`arange` takes a step, `linspace` takes a count (endpoint included), `logspace` spaces points on a log scale.

```python
print(np.arange(0, 1, 0.25))
print(np.linspace(0, 1, 5))   # includes endpoint
print(np.logspace(0, 3, 4))   # 10**0 ... 10**3
```

Output:

```text
[0.   0.25 0.5  0.75]
[0.   0.25 0.5  0.75 1.  ]
[   1.   10.  100. 1000.]
```

### 3. Build arrays from a function

`fromfunction` calls your function with index grids, handy for coordinate-based patterns.

```python
grid = np.fromfunction(lambda i, j: i * 3 + j, (3, 4), dtype=int)
print(grid)
```

Output:

```text
[[0 1 2 3]
 [3 4 5 6]
 [6 7 8 9]]
```

### 4. Structured arrays (mini tables)

Mixed-type records with named fields, no pandas required.

```python
dt = np.dtype([("name", "U10"), ("age", "i4"), ("score", "f8")])
people = np.array([("Asha", 25, 88.5), ("Rahim", 31, 72.0), ("Mitu", 28, 91.2)], dtype=dt)
print(people["name"])
print(people[people["age"] > 26])
print(np.sort(people, order="score")[::-1]["name"])
```

Output:

```text
['Asha' 'Rahim' 'Mitu']
[('Rahim', 31, 72. ) ('Mitu', 28, 91.2)]
['Mitu' 'Asha' 'Rahim']
```

### 5. diag, triu and tri

Build or extract diagonal and triangular structures.

```python
print(np.diag([1, 2, 3]))
print(np.triu(np.arange(1, 10).reshape(3, 3)))
print(np.tri(3, dtype=int))
```

Output:

```text
[[1 0 0]
 [0 2 0]
 [0 0 3]]
[[1 2 3]
 [0 5 6]
 [0 0 9]]
[[1 0 0]
 [1 1 0]
 [1 1 1]]
```

### 6. Integer overflow and safe widening

Fixed-width integers wrap around silently. Widen the dtype before arithmetic when values may exceed the limit.

```python
x = np.array([200], dtype=np.uint8)
print(x + 100)                     # wraps around silently
print(np.iinfo(np.uint8).max)
print(x.astype(np.int16) + 100)    # widen first to be safe
```

Output:

```text
[44]
255
[300]
```

## Indexing & Selection

Source file: [`examples/02_indexing_selection.py`](examples/02_indexing_selection.py)

### 7. Views vs copies

Basic slices return views (shared memory); fancy indexing returns copies. Mixing these up causes subtle bugs.

```python
a = np.arange(6)
view = a[1:4]
view[0] = 99
print(a)
print(np.shares_memory(a, view))
copy = a[[1, 2, 3]]            # fancy indexing -> copy
print(np.shares_memory(a, copy))
```

Output:

```text
[ 0 99  2  3  4  5]
True
False
```

### 8. Fancy indexing with index arrays

Two index arrays select element pairs; chain indexing to pick a sub-grid.

```python
m = np.arange(12).reshape(3, 4)
print(m)
print(m[[0, 2], [1, 3]])       # pairs: (0,1) and (2,3)
print(m[[0, 2]][:, [1, 3]])    # rows 0,2 then columns 1,3
```

Output:

```text
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[ 1 11]
[[ 1  3]
 [ 9 11]]
```

### 9. np.ix_ for sub-grids

`np.ix_` builds an open mesh so you can pick rows and columns in one step.

```python
m = np.arange(12).reshape(3, 4)
print(m[np.ix_([0, 2], [1, 3])])
```

Output:

```text
[[ 1  3]
 [ 9 11]]
```

### 10. Boolean masks with multiple conditions

Combine conditions with `&`, `|`, `~` (not `and`/`or`) and wrap each in parentheses.

```python
a = np.arange(20)
mask = (a % 2 == 0) & (a > 10)
print(mask.sum())
print(a[mask])
print(a[~mask][:5])
```

Output:

```text
4
[12 14 16 18]
[0 1 2 3 4]
```

### 11. argwhere and nonzero

Get the coordinates of elements that satisfy a condition.

```python
m = np.array([[1, 0, 3], [0, 5, 0]])
print(np.argwhere(m > 0))
rows, cols = np.nonzero(m)
print(rows, cols)
```

Output:

```text
[[0 0]
 [0 2]
 [1 1]]
[0 0 1] [0 2 1]
```

### 12. take_along_axis with argsort

Reorder each row independently using indices from `argsort`.

```python
m = np.array([[3, 1, 2], [9, 7, 8]])
idx = np.argsort(m, axis=1)
print(idx)
print(np.take_along_axis(m, idx, axis=1))
```

Output:

```text
[[1 2 0]
 [1 2 0]]
[[1 2 3]
 [7 8 9]]
```

### 13. newaxis and Ellipsis

Add dimensions with `None`/`np.newaxis`, and use `...` to skip leading axes.

```python
x = np.arange(3)
print(x[:, np.newaxis].shape, x[None, :].shape)
t = np.zeros((2, 3, 4))
print(t[..., 0].shape)
print(t[0, ...].shape)
```

Output:

```text
(3, 1) (1, 3)
(2, 3)
(3, 4)
```

### 14. np.select for multi-branch conditions

Vectorised if / elif / else. The first matching condition wins.

```python
scores = np.array([95, 72, 58, 81, 66])
grades = np.select([scores >= 80, scores >= 60], ["A", "B"], default="C")
print(grades)
```

Output:

```text
['A' 'B' 'C' 'A' 'B']
```

## Shape Manipulation

Source file: [`examples/03_shape_manipulation.py`](examples/03_shape_manipulation.py)

### 15. reshape with -1 and memory order

`-1` infers a dimension. `order='F'` reads column-major like Fortran/MATLAB.

```python
a = np.arange(12)
print(a.reshape(3, -1).shape)
print(a.reshape(2, 3, 2).shape)
print(a.reshape(3, 4, order="F")[:, 0])
```

Output:

```text
(3, 4)
(2, 3, 2)
[0 1 2]
```

### 16. stack along a new axis

Unlike `concatenate`, `stack` creates a new dimension.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.stack([a, b], axis=0))
print(np.stack([a, b], axis=1))
```

Output:

```text
[[1 2 3]
 [4 5 6]]
[[1 4]
 [2 5]
 [3 6]]
```

### 17. moveaxis and swapaxes

Convert between HWC and CHW image layouts without copying data.

```python
img = np.zeros((4, 5, 3))      # H, W, C
chw = np.moveaxis(img, -1, 0)
print(chw.shape)
print(np.swapaxes(img, 0, 1).shape)
```

Output:

```text
(3, 4, 5)
(5, 4, 3)
```

### 18. Padding arrays

Pad with constants or by repeating edge values, common in image and signal processing.

```python
print(np.pad(np.ones((2, 2), dtype=int), 1, constant_values=0))
print(np.pad([1, 2, 3], (2, 1), mode="edge"))
```

Output:

```text
[[0 0 0 0]
 [0 1 1 0]
 [0 1 1 0]
 [0 0 0 0]]
[1 1 1 2 3 3]
```

### 19. roll and flip

Circular shifts and reversals along an axis.

```python
a = np.arange(1, 6)
print(np.roll(a, 2))
print(np.flip(a))
m = np.arange(6).reshape(2, 3)
print(np.flip(m, axis=1))
```

Output:

```text
[4 5 1 2 3]
[5 4 3 2 1]
[[2 1 0]
 [5 4 3]]
```

### 20. np.block for block matrices

Assemble a large array from smaller blocks using nested lists.

```python
A = np.eye(2, dtype=int)
Z = np.zeros((2, 2), dtype=int)
print(np.block([[A, Z], [Z, 2 * A]]))
```

Output:

```text
[[1 0 0 0]
 [0 1 0 0]
 [0 0 2 0]
 [0 0 0 2]]
```

## Broadcasting & Ufuncs

Source file: [`examples/04_broadcasting_ufuncs.py`](examples/04_broadcasting_ufuncs.py)

### 21. Pairwise distance matrix without loops

Insert axes so shapes broadcast to (n, n, d), then reduce.

```python
pts = np.array([[0, 0], [3, 4], [6, 8]])
diff = pts[:, None, :] - pts[None, :, :]
dist = np.sqrt((diff ** 2).sum(axis=-1))
print(dist)
```

Output:

```text
[[ 0.  5. 10.]
 [ 5.  0.  5.]
 [10.  5.  0.]]
```

### 22. Column-wise z-score with keepdims

`keepdims=True` keeps reduced axes as size 1 so broadcasting just works.

```python
X = np.array([[1., 200.], [2., 400.], [3., 600.]])
Z = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
print(np.round(Z, 3))
```

Output:

```text
[[-1.225 -1.225]
 [ 0.     0.   ]
 [ 1.225  1.225]]
```

### 23. ufunc reduce, accumulate and outer

Every binary ufunc has `.reduce`, `.accumulate` and `.outer` methods.

```python
a = np.arange(1, 5)
print(np.add.reduce(a), np.multiply.reduce(a))
print(np.add.accumulate(a))
print(np.multiply.outer(np.arange(1, 4), np.arange(1, 4)))
```

Output:

```text
10 24
[ 1  3  6 10]
[[1 2 3]
 [2 4 6]
 [3 6 9]]
```

### 24. np.add.at for repeated indices

`a[idx] += 1` ignores repeated indices. `ufunc.at` handles them correctly.

```python
idx = np.array([0, 0, 2, 4, 4, 4])
counts = np.zeros(5, dtype=int)
np.add.at(counts, idx, 1)
print(counts)
counts2 = np.zeros(5, dtype=int)
counts2[idx] += 1
print(counts2)    # buffered: repeats are lost
```

Output:

```text
[2 0 1 0 3]
[1 0 1 0 1]
```

### 25. np.vectorize (convenience, not speed)

Wraps a scalar Python function so it accepts arrays. It is still a Python loop underneath.

```python
def collatz_step(n):
    return n // 2 if n % 2 == 0 else 3 * n + 1

step = np.vectorize(collatz_step)
print(step(np.arange(1, 8)))
```

Output:

```text
[ 4  1 10  2 16  3 22]
```

### 26. einsum for tensor operations

One notation for matmul, transpose, trace and batched products.

```python
A = np.arange(6).reshape(2, 3)
B = np.arange(6).reshape(3, 2)
print(np.einsum("ij,jk->ik", A, B))
print(np.einsum("ij->ji", A).shape)
print(np.einsum("ii", np.eye(3)))
batch = np.ones((4, 2, 3))
print(np.einsum("bij,bjk->bik", batch, np.ones((4, 3, 5))).shape)
```

Output:

```text
[[10 13]
 [28 40]]
(3, 2)
3.0
(4, 2, 5)
```

### 27. clip with in-place output

Bound values to a range; `out=` avoids allocating a new array.

```python
a = np.array([-5, 0, 5, 10, 15])
print(np.clip(a, 0, 10))
np.clip(a, 0, 10, out=a)
print(a)
```

Output:

```text
[ 0  0  5 10 10]
[ 0  0  5 10 10]
```

## Statistics & Data Analysis

Source file: [`examples/05_statistics_data_analysis.py`](examples/05_statistics_data_analysis.py)

### 28. Percentiles and IQR outlier detection

Median and percentiles resist outliers better than the mean.

```python
data = np.array([12, 15, 11, 18, 21, 14, 90])
print(np.median(data), data.mean())
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
print(data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)])
```

Output:

```text
15.0 25.857142857142858
[90]
```

### 29. Correlation matrix

`corrcoef` returns the full Pearson correlation matrix.

```python
rng = np.random.default_rng(0)
x = rng.normal(size=100)
y = 2 * x + rng.normal(scale=0.5, size=100)
print(np.round(np.corrcoef(x, y), 3))
```

Output:

```text
[[1.    0.971]
 [0.971 1.   ]]
```

### 30. bincount and histogram

`bincount` counts non-negative integers; `histogram` bins continuous values.

```python
rng = np.random.default_rng(1)
data = rng.integers(0, 5, size=20)
print(data)
print(np.bincount(data))
counts, edges = np.histogram(rng.normal(size=1000), bins=5)
print(counts)
print(np.round(edges, 2))
```

Output:

```text
[2 2 3 4 0 0 4 4 1 1 4 2 1 4 1 2 3 2 0 0]
[4 4 5 2 5]
[ 25 256 530 182   7]
[-3.55 -2.09 -0.63  0.83  2.29  3.75]
```

### 31. digitize for binning

Map continuous values to bin indices, then to labels.

```python
ages = np.array([5, 17, 25, 42, 67, 80])
bins = np.array([18, 40, 65])
labels = np.array(["child", "young", "adult", "senior"])
print(labels[np.digitize(ages, bins)])
```

Output:

```text
['child' 'child' 'young' 'adult' 'senior' 'senior']
```

### 32. Weighted average

Use weights, for example credit hours in a GPA calculation.

```python
grades = np.array([80, 90, 70])
credits = np.array([3, 4, 2])
print(np.average(grades, weights=credits))
```

Output:

```text
82.22222222222223
```

### 33. diff and gradient

Discrete differences and numerical derivatives.

```python
y = np.array([1, 4, 9, 16, 25])
print(np.diff(y))
print(np.gradient(y))
```

Output:

```text
[3 5 7 9]
[3. 4. 6. 8. 9.]
```

### 34. Moving average with convolve

Smooth a signal with a uniform kernel; `mode='valid'` drops the edges.

```python
x = np.arange(1, 11, dtype=float)
w = 3
print(np.convolve(x, np.ones(w) / w, mode="valid"))
```

Output:

```text
[2. 3. 4. 5. 6. 7. 8. 9.]
```

### 35. Running maximum

Cumulative versions exist for any ufunc via `.accumulate`.

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(np.maximum.accumulate(a))
```

Output:

```text
[3 3 4 4 5 9 9 9]
```

## Linear Algebra & Fitting

Source file: [`examples/06_linear_algebra_fitting.py`](examples/06_linear_algebra_fitting.py)

### 36. Linear regression with lstsq

Solve least squares directly, no sklearn needed.

```python
rng = np.random.default_rng(7)
x = np.linspace(0, 10, 50)
y = 3.0 * x + 2.0 + rng.normal(scale=0.5, size=50)
X = np.column_stack([x, np.ones_like(x)])
(slope, intercept), *_ = np.linalg.lstsq(X, y, rcond=None)
print(round(slope, 3), round(intercept, 3))
```

Output:

```text
3.019 1.757
```

### 37. SVD and low-rank approximation

Keep only the top singular value to get the best rank-1 approximation.

```python
rng = np.random.default_rng(3)
A = rng.normal(size=(6, 4))
U, s, Vt = np.linalg.svd(A, full_matrices=False)
A1 = (U[:, :1] * s[:1]) @ Vt[:1]
print(np.round(s, 3))
print(np.linalg.matrix_rank(A1))
print(round(np.linalg.norm(A - A1), 3))
```

Output:

```text
[4.888 2.811 1.259 0.388]
1
3.104
```

### 38. eigh for symmetric matrices

`eigh` is faster and more stable than `eig` for symmetric/Hermitian matrices.

```python
S = np.array([[2., 1.], [1., 2.]])
vals, vecs = np.linalg.eigh(S)
print(vals)
print(np.allclose(S @ vecs, vecs * vals))
```

Output:

```text
[1. 3.]
True
```

### 39. Vector and matrix norms

L1, L2, infinity and Frobenius norms in one function.

```python
v = np.array([3., 4.])
print(np.linalg.norm(v), np.linalg.norm(v, 1), np.linalg.norm(v, np.inf))
M = np.array([[1., 2.], [3., 4.]])
print(np.linalg.norm(M), np.linalg.norm(M, axis=1))
```

Output:

```text
5.0 7.0 4.0
5.477225575051661 [2.23606798 5.        ]
```

### 40. matrix_power and Fibonacci

Fibonacci numbers fall out of powers of [[1, 1], [1, 0]].

```python
Fib = np.array([[1, 1], [1, 0]])
p = np.linalg.matrix_power(Fib, 10)
print(p)
print(p[0, 1])
```

Output:

```text
[[89 55]
 [55 34]]
55
```

### 41. Polynomial fit and interpolation

`polyfit` fits a polynomial; `interp` does piecewise-linear interpolation.

```python
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 7, 13, 21])      # y = x^2 + x + 1
coef = np.polyfit(x, y, 2)
print(np.round(coef, 3))
print(round(float(np.polyval(coef, 5)), 3))
print(np.interp(2.5, x, y))
```

Output:

```text
[1. 1. 1.]
31.0
10.0
```

## Random Numbers (Generator API)

Source file: [`examples/07_random_numbers_generator_api.py`](examples/07_random_numbers_generator_api.py)

### 42. default_rng basics

Prefer the modern `Generator` API over legacy `np.random.*` functions.

```python
rng = np.random.default_rng(42)
print(rng.integers(1, 100, size=5))
print(np.round(rng.normal(loc=50, scale=10, size=3), 2))
print(np.round(rng.random(3), 3))
```

Output:

```text
[ 9 77 65 44 43]
[59.41 30.49 36.98]
[0.761 0.786 0.128]
```

### 43. permutation, choice and shuffle

Sample without replacement, or shuffle in place.

```python
rng = np.random.default_rng(0)
a = np.arange(10)
print(rng.permutation(a))
print(rng.choice(a, size=4, replace=False))
rng.shuffle(a)    # in-place
print(a)
```

Output:

```text
[4 6 2 7 3 5 9 0 8 1]
[3 7 4 8]
[6 4 2 5 1 3 9 8 0 7]
```

### 44. Monte Carlo estimate of pi

Vectorised simulation of a million points in one go.

```python
rng = np.random.default_rng(0)
pts = rng.random((1_000_000, 2))
inside = (pts ** 2).sum(axis=1) <= 1
print(4 * inside.mean())
```

Output:

```text
3.141168
```

### 45. Bootstrap confidence interval

Resample with replacement thousands of times, entirely vectorised.

```python
rng = np.random.default_rng(5)
sample = rng.normal(loc=100, scale=15, size=50)
boot = rng.choice(sample, size=(2000, sample.size), replace=True).mean(axis=1)
print(np.round(np.percentile(boot, [2.5, 97.5]), 2))
```

Output:

```text
[91.58 98.69]
```

## Practical Patterns

Source file: [`examples/08_practical_patterns.py`](examples/08_practical_patterns.py)

### 46. sliding_window_view

Zero-copy rolling windows, no manual loops.

```python
from numpy.lib.stride_tricks import sliding_window_view

x = np.arange(1, 8)
win = sliding_window_view(x, 3)
print(win)
print(win.mean(axis=1))
```

Output:

```text
[[1 2 3]
 [2 3 4]
 [3 4 5]
 [4 5 6]
 [5 6 7]]
[2. 3. 4. 5. 6.]
```

### 47. Masked arrays

Exclude sentinel values (like -999) from computations without deleting them.

```python
data = np.ma.masked_values([1.0, -999.0, 3.0, -999.0, 5.0], -999.0)
print(data)
print(data.mean())
print(data.filled(0))
```

Output:

```text
[1.0 -- 3.0 -- 5.0]
3.0
[1. 0. 3. 0. 5.]
```

### 48. NaN handling and mean imputation

Use `nan*` functions and `where` to fill missing values per column.

```python
X = np.array([[1., 2.], [np.nan, 4.], [5., np.nan]])
col_mean = np.nanmean(X, axis=0)
filled = np.where(np.isnan(X), col_mean, X)
print(col_mean)
print(filled)
```

Output:

```text
[3. 3.]
[[1. 2.]
 [3. 4.]
 [5. 3.]]
```

### 49. Stable softmax and one-hot encoding

Subtract the max before `exp` to avoid overflow. Index an identity matrix for one-hot labels.

```python
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

print(np.round(softmax(np.array([1000., 1001., 1002.])), 3))
labels = np.array([0, 2, 1, 2])
print(np.eye(3, dtype=int)[labels])
```

Output:

```text
[0.09  0.245 0.665]
[[1 0 0]
 [0 0 1]
 [0 1 0]
 [0 0 1]]
```

### 50. Save and load with .npz

Store several arrays in one compressed file.

```python
import os, tempfile

path = os.path.join(tempfile.gettempdir(), "demo.npz")
np.savez_compressed(path, a=np.arange(5), b=np.eye(2))
loaded = np.load(path)
print(loaded.files)
print(loaded["a"], loaded["b"].shape)
```

Output:

```text
['a', 'b']
[0 1 2 3 4] (2, 2)
```

## Contributing

Found a bug or have a useful NumPy trick? Open an issue or send a pull request. Please keep each example short, self-contained and deterministic (fixed seeds).
