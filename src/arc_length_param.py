"""Arc-length parameterization of a sampled closed 1D manifold and its natural modes."""
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.integrate import cumulative_trapezoid
import arc_length_param_viz
from arc_length_param_viz import serve

B = 2
K = 3 
N = 200

# 1. Define functions and sample point cloud
def r(theta):
    return 2 + B * np.cos(K * theta)
    
def x(theta):
    return r(theta) * np.cos(theta)

def y(theta):
    return r(theta) * np.sin(theta)
    
def ambient_point(theta):
    return (x(theta), y(theta))

def sample_manifold(start, end = 2 * np.pi, n=N, eps = 0.5, rng=None):
    rng = np.random.default_rng(rng)
    grid = np.linspace(start, end, n, endpoint = False)
    rand = np.sort(rng.uniform(start, end, n))
    theta = (1 - eps) * grid + eps * rand
    return np.column_stack(ambient_point(theta)), theta 
    
# 2. Cumulative chord length
def cumulative_chord_length(points):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.r_[0, np.cumsum(seg)]
    return np.column_stack((t, points))

# 3. Fit continuous vector valued B-spline through pairs
def b_spline(knots):
    return make_interp_spline(knots[:, 0], knots[:, 1:])

# 4. Compute arc length along fitted curve
def arclength_from_cumulative_chords(spline, t):
    v = np.linalg.norm(spline.derivative()(t), axis=-1)
    return cumulative_trapezoid(v, t, initial=0)
    
# 5. Find the warped fourier bases
def phi_k(k, warp, theta):
    L = warp[-1]
    true_fourier = np.exp(2j * np.pi * k * theta / (2 * np.pi))
    warped_fourier = np.exp(2j * np.pi * k * warp / L)
    return true_fourier, warped_fourier
    
# 6. Define a function on the manifold. Temperature as a function of x and y
def temp(xy):
    """Hot blob on one outer petal: flat at the cusps, not a trig polynomial in θ."""
    x, y = xy[:, 0], xy[:, 1]
    return 10 + 15 * np.exp(-((x - 3.2) ** 2 + y ** 2) / (2 * 1.1 ** 2))

def resample_f_pairs(pairs, func, n=N):
    """Evaluate f on a uniform arc-length grid of n points."""
    L = pairs[-1, 0]
    si = np.arange(n) * L / n
    return np.ravel(func(si))


def truncate_rfft(F, k_max=8):
    out = np.zeros_like(F)
    out[: k_max + 1] = F[: k_max + 1]
    return out


def fft_approx_on_samples(s_hat, f_fft, n, L):
    """Inverse rFFT on the uniform ŝ grid, then interpolate back to sample ŝ."""
    s_grid = np.arange(n) * L / n
    f_grid = np.fft.irfft(f_fft, n=n)
    return np.interp(s_hat, s_grid, f_grid, period=L)


def recon_error(approx, f):
    err = np.abs(approx - f)
    return float(np.sqrt(np.mean(err ** 2))), float(err.max())
     
      
def run():
    points, theta = sample_manifold(0, eps=0)
    # Show manifold
    arc_length_param_viz.show_manifold(points)
    
    # 2. Calculate cumulative chord lengths
    t = cumulative_chord_length(points)
    spline = b_spline(t)
    s_hat = arclength_from_cumulative_chords(spline, t[:, 0])
    modes = {k: phi_k(k, s_hat, theta) for k in range(1, 9)}
    arc_length_param_viz.show_spline(spline, t)
    arc_length_param_viz.show_warp(theta, s_hat)
    arc_length_param_viz.show_modes(theta, modes)
    
    # 6. Approximate Function on Manifold
    f = temp(points)
    f_pairs = np.column_stack([s_hat, f])
    f_spline = b_spline(f_pairs)
    vals = resample_f_pairs(f_pairs, f_spline)

    f_fft = truncate_rfft(np.fft.rfft(vals))
    f_hat = fft_approx_on_samples(s_hat, f_fft, len(vals), f_pairs[-1, 0])

    f_reg = np.fft.irfft(truncate_rfft(np.fft.rfft(f)), n=len(f))
    warp_rmse, warp_max = recon_error(f_hat, f)
    reg_rmse, reg_max = recon_error(f_reg, f)
    print(f"warped FFT  RMSE={warp_rmse:.4f}  max|err|={warp_max:.4f}")
    print(f"regular FFT RMSE={reg_rmse:.4f}  max|err|={reg_max:.4f}", flush=True)
    arc_length_param_viz.show_function(f, f_hat, f_reg, dict(
        warp_rmse=warp_rmse, warp_max=warp_max, reg_rmse=reg_rmse, reg_max=reg_max,
    ))
    serve()


if __name__ == "__main__":
    run()