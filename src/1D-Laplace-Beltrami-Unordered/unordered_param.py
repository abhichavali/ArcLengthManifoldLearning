"""Unordered sampling of the closed 1D manifold: random θ, no sorting, shuffled indices."""
import os, sys, time
import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1D-Laplace-Beltrami"))
from arc_length_param import temp, recon_error, K, N
from arc_length_vs_d_map import diffusion_kernel, diffusion_map_fit, arc_length_fit, bench, N_BASIS
import unordered_viz

SEED = 0
B = 1  # not 2: r = 2 + 2cos3θ passes through the origin 3x, so petals are ambient-close there
       # and no kernel can order the points near the cusps. r = 2 + cos3θ >= 1 is embedded.


def ambient_point(theta):
    r = 2 + B * np.cos(K * theta)
    return r * np.cos(theta), r * np.sin(theta)


def sample_unordered(n=N, rng=SEED):
    """n points at θ ~ U(0, 2π) in the order drawn. Returns (points, theta)."""
    rng = np.random.default_rng(rng)
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack(ambient_point(theta)), theta


def diffusion_embedding(points, eps=None):
    """First 2 nontrivial diffusion coordinates (λ₁ψ₁, λ₂ψ₂) via a partial eigensolve."""
    S, d = diffusion_kernel(points, eps)
    lam, U = eigsh(S, k=3, which="LA", v0=np.ones(len(points)))  # Lanczos: top 3 of S; fixed start => deterministic
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order] / np.sqrt(d)[:, None]  # right eigenvectors of P
    return U[:, 1:3] * lam[1:3]              # drop the constant ψ₀


def recover_order(embedding):
    """Rank of each point around the embedded circle. rank[i] = position of sample i."""
    angle = np.arctan2(embedding[:, 1], embedding[:, 0])
    return np.argsort(np.argsort(angle)), angle


def order_points(points):
    """Unordered cloud -> (points sorted along the curve, rank of each original point, angle)."""
    rank, angle = recover_order(diffusion_embedding(points))
    return points[np.argsort(rank)], rank, angle


def closed_arc_length_fit(points, f):
    """arc_length_fit on the loop closed back to its first point, so L is the full circumference
    and the FFT's periodization has no jump at the seam (random sampling can leave a big last gap)."""
    return arc_length_fit(np.vstack([points, points[:1]]), np.r_[f, f[0]], "linear")[:-1]


def sweep(ns=(200, 400, 800, 1600, 3200), repeats=3):
    """Per N: (N, order-recovery ms, arc-length fit ms, full diffusion ms, arc RMSE, diffusion RMSE)."""
    rows = []
    for n in ns:
        p, _ = sample_unordered(n); g = temp(p)
        _, t_order, _ = bench(order_points, p, repeats=repeats)
        ps, rank, _ = order_points(p)
        a, t_arc, _ = bench(closed_arc_length_fit, ps, g[np.argsort(rank)], repeats=repeats)
        d, t_dm, _ = bench(diffusion_map_fit, p, g, repeats=repeats)
        rows.append((n, t_order, t_arc, t_dm, recon_error(a, g[np.argsort(rank)])[0], recon_error(d, g)[0]))
    return rows


def run():
    points, theta = sample_unordered()
    assert np.any(np.diff(theta) < 0)  # genuinely unordered
    sorted_points, rank, angle = order_points(points)
    # walking the points in recovered order, θ must be monotone (either orientation) up to one wrap
    d = np.diff(theta[np.argsort(rank)]); d = d[np.abs(d) < np.pi]
    assert np.all(d > 0) or np.all(d < 0), "recovered order has local swaps"
    unordered_viz.show_samples(points)
    unordered_viz.show_ordered(points, rank, angle)

    # Same pipeline as 1D-Laplace-Beltrami on the recovered ordering:
    # chord lengths -> B-spline -> arc length -> resample -> rFFT (k<=16) -> reconstruct.
    f = temp(sorted_points)
    f_hat = closed_arc_length_fit(sorted_points, f)
    print(f"arc-length on recovered order: RMSE={recon_error(f_hat, f)[0]:.4f}  ({N_BASIS} coefficients)")
    unordered_viz.show_benchmark(sweep())
    unordered_viz.serve()


if __name__ == "__main__":
    run()
