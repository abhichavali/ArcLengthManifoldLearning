"""Time arc-length Fourier vs diffusion maps at approximating temp() on the same point cloud."""
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform

import arc_length_param as alp

K_MAX = 16
N_BASIS = 2 * K_MAX + 1  # DC + 16 cos + 16 sin == 33 real coefficients
REPEATS = 20


def arc_length_fit(points, f, resample="spline"):
    t = alp.cumulative_chord_length(points)
    s_hat = alp.arclength_from_cumulative_chords(alp.b_spline(t), t[:, 0])
    f_pairs = np.column_stack([s_hat, f])
    if resample == "linear":  # interpolating splines oscillate on near-duplicate s (random sampling)
        vals = np.interp(np.arange(len(f)) * s_hat[-1] / len(f), s_hat, f)
    else:
        vals = alp.resample_f_pairs(f_pairs, alp.b_spline(f_pairs))
    f_fft = alp.truncate_rfft(np.fft.rfft(vals), K_MAX)
    return alp.fft_approx_on_samples(s_hat, f_fft, len(vals), f_pairs[-1, 0])


def diffusion_kernel(points, eps=None):
    """Symmetric conjugate S = D^-1/2 W D^-1/2 of the alpha=1 Markov matrix, and its degrees d."""
    D2 = squareform(pdist(points, "sqeuclidean"))
    eps = eps or np.median(D2[D2 > 0])  # ponytail: median heuristic, tune if needed
    W = np.exp(-D2 / eps)
    q = W.sum(1)
    W = W / np.outer(q, q)              # alpha=1 (Laplace-Beltrami) normalization
    d = W.sum(1)
    return W / np.sqrt(np.outer(d, d)), d


def diffusion_map_fit(points, f, eps=None, n_basis=N_BASIS):
    S, d = diffusion_kernel(points, eps)
    lam, U = np.linalg.eigh(S)
    U = U[:, ::-1][:, :n_basis] / np.sqrt(d)[:, None]  # right eigenvectors of P
    # eigenvectors of P are orthonormal in the d-weighted inner product
    coef = U.T @ (d * f) / (U.T @ (d[:, None] * U)).diagonal()
    return U @ coef


def bench(fn, *args, repeats=REPEATS):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(*args); ts.append(time.perf_counter() - t0)
    return out, np.median(ts) * 1e3, np.min(ts) * 1e3


def sweep(ns=(200, 400, 800, 1600, 3200), repeats=REPEATS):
    """Per N: (N, arc ms, diffusion ms, arc RMSE, diffusion RMSE)."""
    rows = []
    for n in ns:
        p, _ = alp.sample_manifold(0, n=n, eps=0); g = alp.temp(p)
        a, a_med, _ = bench(arc_length_fit, p, g, repeats=repeats)
        d, d_med, _ = bench(diffusion_map_fit, p, g, repeats=repeats)
        rows.append((n, a_med, d_med, alp.recon_error(a, g)[0], alp.recon_error(d, g)[0]))
    return rows


if __name__ == "__main__":
    points, theta = alp.sample_manifold(0, eps=0)
    f = alp.temp(points)
    rows = []
    for name, fn in [("arc-length FFT", arc_length_fit), ("diffusion maps", diffusion_map_fit)]:
        approx, med, best = bench(fn, points, f)
        rmse, mx = alp.recon_error(approx, f)
        rows.append((name, med, best, rmse, mx))
        print(f"{name:15s} median={med:8.3f} ms  min={best:8.3f} ms  RMSE={rmse:.4f}  max|err|={mx:.4f}")
    # sanity: same 17-term budget, both should beat the trivial mean
    assert all(r[3] < f.std() for r in rows)
    # scaling: diffusion maps is O(N^3) in the eigensolve, arc-length is O(N log N)
    print()
    for n, a_med, d_med, _, _ in sweep():
        print(f"N={n:5d}  arc-length={a_med:8.3f} ms  diffusion={d_med:9.3f} ms  ratio={d_med/a_med:7.1f}x")
