# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Visualize a 1D manifold in R^2, a function on it, and its natural modes.

Run: uv run experiments/viz_1d_manifold.py
Writes experiments/viz_1d_manifold.png
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
N, K = 400, 4  # samples, modes shown

# --- closed curve, sampled non-uniformly in t so arc length != t ---
t = np.sort(rng.uniform(0, 2 * np.pi, N))
X = np.c_[np.cos(t) + 0.3 * np.cos(3 * t), np.sin(t) + 0.3 * np.sin(2 * t)]

# --- arc length: cumulative chord length (closed: wrap last->first) ---
seg = np.linalg.norm(np.diff(X, axis=0, append=X[:1]), axis=1)
s = np.r_[0, np.cumsum(seg[:-1])]
L = seg.sum()

# --- natural modes on a closed 1D manifold = Fourier modes in arc length ---
def mode(k):  # k=0 const, odd -> cos, even -> sin, ordered by eigenvalue
    m = (k + 1) // 2
    return np.cos(2 * np.pi * m * s / L) if k % 2 else np.sin(2 * np.pi * m * s / L)

# --- a function on the manifold and its spectrum via FFT on an arc-length grid ---
f = np.sin(2 * np.pi * 2 * s / L) + 0.5 * np.cos(2 * np.pi * 5 * s / L) + 0.1 * rng.standard_normal(N)
s_grid = np.linspace(0, L, N, endpoint=False)
f_grid = np.interp(s_grid, s, f, period=L)  # ponytail: linear interp; spline if aliasing matters
spec = np.abs(np.fft.rfft(f_grid)) / N * 2

# --- comparison: graph Laplacian (Gaussian kernel) eigenvectors, i.e. diffusion-map style ---
D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
W = np.exp(-D2 / (2 * seg.max()) ** 2)  # bandwidth ~ largest gap so the graph stays connected
Lap = np.diag(W.sum(1)) - W
_, V = np.linalg.eigh(Lap)

# --- plot ---
fig, ax = plt.subplots(3, K + 1, figsize=(3.2 * (K + 1), 9))
sc = lambda a, c, title, sym=False: (a.scatter(*X.T, c=c, s=8, cmap="coolwarm", **({"vmin": -abs(c).max(), "vmax": abs(c).max()} if sym else {})), a.set_aspect("equal"), a.set_title(title, fontsize=9), a.axis("off"))

sc(ax[0, 0], s, "manifold, colored by arc length s")
sc(ax[0, 1], f, "function f on manifold")
ax[0, 2].plot(s, f, ".", ms=3); ax[0, 2].set(title="f vs arc length s", xlabel="s")
ax[0, 3].stem(spec[:15]); ax[0, 3].set(title="|FFT(f)| in arc length", xlabel="mode m")
ax[0, 4].axis("off")

for k in range(K + 1):
    sc(ax[1, k], mode(k), f"arc-length mode {k}", sym=True)
    sc(ax[2, k], V[:, k], f"graph Laplacian eigvec {k}", sym=True)  # symmetric scale so a constant mode is not float noise

fig.tight_layout()
out = __file__.replace(".py", ".png")
fig.savefig(out, dpi=110)
print("wrote", out)

if __name__ == "__main__":
    # self-check: arc-length modes should recover the exact frequencies planted in f
    assert spec[2] > 0.8 and spec[5] > 0.4 and spec[3] < 0.2, spec[:8]
    assert abs(np.abs(np.corrcoef(mode(1), V[:, 1])[0, 1]) - 1) < 0.2 or abs(np.abs(np.corrcoef(mode(2), V[:, 1])[0, 1]) - 1) < 0.2
