"""Time arc-length Fourier vs diffusion maps at approximating temp() on the same point cloud."""
import os, sys, time
import numpy as np
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.modules["arc_length_param_viz"] = type(sys)("arc_length_param_viz")  # skip plotly import
sys.modules["arc_length_param_viz"].serve = lambda: None
import arc_length_param as alp

K_MAX = 8
N_BASIS = 2 * K_MAX + 1  # DC + 8 cos + 8 sin == 17 real coefficients
REPEATS = 20


def arc_length_fit(points, f):
    t = alp.cumulative_chord_length(points)
    s_hat = alp.arclength_from_cumulative_chords(alp.b_spline(t), t[:, 0])
    f_pairs = np.column_stack([s_hat, f])
    vals = alp.resample_f_pairs(f_pairs, alp.b_spline(f_pairs))
    f_fft = alp.truncate_rfft(np.fft.rfft(vals), K_MAX)
    return alp.fft_approx_on_samples(s_hat, f_fft, len(vals), f_pairs[-1, 0])


def diffusion_map_fit(points, f, eps=None, n_basis=N_BASIS):
    D2 = squareform(pdist(points, "sqeuclidean"))
    eps = eps or np.median(D2[D2 > 0])  # ponytail: median heuristic, tune if needed
    W = np.exp(-D2 / eps)
    q = W.sum(1)
    W = W / np.outer(q, q)              # alpha=1 (Laplace-Beltrami) normalization
    d = W.sum(1)
    S = W / np.sqrt(np.outer(d, d))     # symmetric conjugate of the Markov matrix
    lam, U = np.linalg.eigh(S)
    U = U[:, ::-1][:, :n_basis] / np.sqrt(d)[:, None]  # right eigenvectors of P
    # eigenvectors of P are orthonormal in the d-weighted inner product
    coef = U.T @ (d * f) / (U.T @ (d[:, None] * U)).diagonal()
    return U @ coef


def bench(fn, *args):
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter(); out = fn(*args); ts.append(time.perf_counter() - t0)
    return out, np.median(ts) * 1e3, np.min(ts) * 1e3


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
    for n in (200, 400, 800, 1600, 3000):
        p, _ = alp.sample_manifold(0, n=n, eps=0); g = alp.temp(p)
        _, a_med, _ = bench(arc_length_fit, p, g)
        _, d_med, _ = bench(diffusion_map_fit, p, g)
        print(f"N={n:5d}  arc-length={a_med:8.3f} ms  diffusion={d_med:9.3f} ms  ratio={d_med/a_med:7.1f}x")
