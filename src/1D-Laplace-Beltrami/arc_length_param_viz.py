"""Web visualizer for the sampled 1D manifold. Run: python src/arc_length_param.py"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HOST, PORT = "127.0.0.1", 8000
_points = None
_spline = None
_knots = None
_t = None
_s_hat = None
_modes = None
_f = None
_f_hat = None
_f_reg = None
_f_stats = None

HTML = """<!doctype html>
<meta charset="utf-8">
<title>Arc length manifold</title>
<style>
  body { font: 16px/1.4 system-ui, sans-serif; margin: 0; }
  nav { display: flex; gap: .5rem; padding: .75rem 1rem; border-bottom: 1px solid #ccc; }
  nav button { padding: .4rem .8rem; cursor: pointer; }
  nav button.active { font-weight: 700; }
  .panel { display: none; }
  .panel.active { display: block; }
</style>
<nav>
  <button class="active" data-tab="1">1. Parametric curve</button>
  <button data-tab="2">2. B-spline</button>
  <button data-tab="3">3. Arc-length warp</button>
  <button data-tab="4">4. Fourier modes</button>
  <button data-tab="5">5. Complex plane</button>
  <button data-tab="6">6. Temperature</button>
</nav>
<section class="panel active" data-tab="1"><!--PLOT1--></section>
<section class="panel" data-tab="2"><!--PLOT2--></section>
<section class="panel" data-tab="3"><!--PLOT3--></section>
<section class="panel" data-tab="4"><!--PLOT4--></section>
<section class="panel" data-tab="5"><!--PLOT5--></section>
<section class="panel" data-tab="6"><!--PLOT6--></section>
<script>
document.querySelector("nav").onclick = e => {
  const t = e.target.dataset.tab;
  if (!t) return;
  document.querySelectorAll("[data-tab]").forEach(el => el.classList.toggle("active", el.dataset.tab === t));
  document.querySelectorAll(".plotly-graph-div").forEach(gd => Plotly.Plots.resize(gd));
};
</script>
"""


def show_manifold(points):
    global _points
    _points = np.asarray(points)


def show_spline(spline, knots=None):
    global _spline, _knots
    _spline = spline
    _knots = None if knots is None else np.asarray(knots)


def show_warp(t, s_hat):
    global _t, _s_hat
    _t = np.asarray(t)
    _s_hat = np.asarray(s_hat)


def show_modes(theta, modes):
    global _modes
    _modes = (np.asarray(theta), {k: (np.asarray(t), np.asarray(w)) for k, (t, w) in modes.items()})


def show_function(f, f_hat=None, f_reg=None, stats=None):
    global _f, _f_hat, _f_reg, _f_stats
    _f = np.asarray(f)
    _f_hat = None if f_hat is None else np.asarray(f_hat)
    _f_reg = None if f_reg is None else np.asarray(f_reg)
    _f_stats = stats


def curve_figure(points):
    xy = np.vstack([points, points[:1]])
    fig = go.Figure(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="lines+markers", marker=dict(size=4)))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(title="γ(θ) in R²", margin=dict(l=40, r=20, t=40, b=40), height=700)
    return fig


def spline_figure(spline, knots=None):
    t0, t1 = float(spline.t[spline.k]), float(spline.t[-spline.k - 1])
    tau = np.linspace(t0, t1, 500)
    xy = spline(tau)
    fig = go.Figure(go.Scatter3d(x=tau, y=xy[:, 0], z=xy[:, 1], mode="lines", name="B-spline"))
    if knots is not None:
        fig.add_trace(go.Scatter3d(
            x=knots[:, 0], y=knots[:, 1], z=knots[:, 2],
            mode="markers", marker=dict(size=3), name="(tᵢ, Pᵢ)",
        ))
    fig.update_layout(
        title="γ̂(t) in (t, x, y)", height=700, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(xaxis_title="t", yaxis_title="x", zaxis_title="y", aspectmode="data"),
    )
    return fig


def warp_figure(theta, s_hat):
    lin = s_hat[-1] * (theta - theta[0]) / (theta[-1] - theta[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theta, y=lin, mode="lines", name="constant speed", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=theta, y=s_hat, mode="lines+markers", marker=dict(size=4), name="ŝ(θ)"))
    fig.update_layout(
        title="arc-length warp ŝ(θ)", xaxis_title="θ", yaxis_title="ŝ",
        height=700, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def modes_figure(theta, modes):
    ks = list(modes)
    k0 = 3 if 3 in modes else ks[0]
    true, warped = modes[k0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theta, y=np.real(true), mode="lines", name="Re e^{ikθ}"))
    fig.add_trace(go.Scatter(x=theta, y=np.real(warped), mode="lines", name="Re e^{2πik ŝ/L}"))
    steps = [dict(
        method="update", label=str(k),
        args=[{"y": [np.real(tr), np.real(wa)]},
              {"title": f"Fourier mode k={k}: regular in θ vs warped in ŝ"}],
    ) for k, (tr, wa) in modes.items()]
    fig.update_layout(
        title=f"Fourier mode k={k0}: regular in θ vs warped in ŝ",
        xaxis_title="θ", yaxis_title="real part",
        height=700, margin=dict(l=40, r=20, t=40, b=80),
        sliders=[dict(active=ks.index(k0), steps=steps, currentvalue=dict(prefix="k = "))],
    )
    return fig


def argand_figure(theta, modes):
    ks = list(modes)
    k0 = 3 if 3 in modes else ks[0]
    true, warped = modes[k0]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["regular e^{ikθ}", "warped e^{2πik ŝ/L}"])
    mk = dict(size=5, color=theta, colorscale="Viridis")
    fig.add_trace(go.Scatter(
        x=np.real(true), y=np.imag(true), mode="lines+markers",
        marker=mk | {"colorbar": dict(title="θ")}, name="regular",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.real(warped), y=np.imag(warped), mode="lines+markers",
        marker=mk | {"showscale": False}, name="warped",
    ), row=1, col=2)
    steps = [dict(
        method="update", label=str(k),
        args=[{"x": [np.real(tr), np.real(wa)], "y": [np.imag(tr), np.imag(wa)]},
              {"title": f"modes in the complex plane, k={k}"}],
    ) for k, (tr, wa) in modes.items()]
    fig.update_xaxes(title_text="Re", row=1, col=1)
    fig.update_xaxes(title_text="Re", row=1, col=2)
    fig.update_yaxes(title_text="Im", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="Im", scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_layout(
        title=f"modes in the complex plane, k={k0}", height=700,
        margin=dict(l=40, r=20, t=60, b=80), showlegend=False,
        sliders=[dict(active=ks.index(k0), steps=steps, currentvalue=dict(prefix="k = "))],
    )
    return fig


def _lift_traces(points, f, name, colorbar=False):
    xy = np.vstack([points, points[:1]])
    z = np.r_[f, f[0]]
    n = len(points)
    gap = np.full(n, np.nan)
    stem_x = np.ravel(np.column_stack([points[:, 0], points[:, 0], gap]))
    stem_y = np.ravel(np.column_stack([points[:, 1], points[:, 1], gap]))
    stem_z = np.ravel(np.column_stack([np.zeros(n), f, gap]))
    mk = dict(size=3, color=z, colorscale="RdBu_r")
    if colorbar:
        mk["colorbar"] = dict(title="T")
    else:
        mk["showscale"] = False
    return [
        go.Scatter3d(x=xy[:, 0], y=xy[:, 1], z=np.zeros(len(xy)), mode="lines", name="manifold", line=dict(color="#888"), showlegend=False),
        go.Scatter3d(x=stem_x, y=stem_y, z=stem_z, mode="lines", name=name, line=dict(color="#bbb", width=1), showlegend=False),
        go.Scatter3d(x=xy[:, 0], y=xy[:, 1], z=z, mode="lines+markers", name=name, marker=mk),
    ]


def function_figure(points, f, f_hat=None, f_reg=None, stats=None):
    scene = dict(xaxis_title="x", yaxis_title="y", zaxis_title="T", aspectmode="data")
    stats = stats or {}
    if f_hat is None:
        fig = go.Figure(_lift_traces(points, f, "T", colorbar=True))
        fig.update_layout(title="T(x, y) lifted off the manifold", height=700, margin=dict(l=0, r=0, t=40, b=0), scene=scene)
        return fig
    if f_reg is None:
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                            subplot_titles=["T on the manifold", "warped FFT reconstruction"])
        for tr in _lift_traces(points, f, "T", colorbar=True):
            fig.add_trace(tr, row=1, col=1)
        for tr in _lift_traces(points, f_hat, "T̂", colorbar=False):
            fig.add_trace(tr, row=1, col=2)
        fig.update_layout(title="T vs warped-FFT approximation", height=700, margin=dict(l=0, r=0, t=40, b=0),
                          scene=scene, scene2=scene, showlegend=False)
        return fig
    warp_t = f"warped FFT (k≤8)<br>RMSE={stats.get('warp_rmse', 0):.3f}  max|err|={stats.get('warp_max', 0):.3f}"
    reg_t = f"regular FFT (k≤8)<br>RMSE={stats.get('reg_rmse', 0):.3f}  max|err|={stats.get('reg_max', 0):.3f}"
    fig = make_subplots(
        rows=1, cols=3, specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["T (original)", warp_t, reg_t],
    )
    for tr in _lift_traces(points, f, "T", colorbar=True):
        fig.add_trace(tr, row=1, col=1)
    for tr in _lift_traces(points, f_hat, "warped", colorbar=False):
        fig.add_trace(tr, row=1, col=2)
    for tr in _lift_traces(points, f_reg, "regular", colorbar=False):
        fig.add_trace(tr, row=1, col=3)
    fig.update_layout(
        title="T vs truncated FFT reconstructions (k = 0…8)", height=740,
        margin=dict(l=0, r=0, t=80, b=0), scene=scene, scene2=scene, scene3=scene, showlegend=False,
    )
    return fig


def page():
    p1 = curve_figure(_points).to_html(full_html=False, include_plotlyjs="cdn")
    p2 = spline_figure(_spline, _knots).to_html(full_html=False, include_plotlyjs=False) if _spline is not None else ""
    p3 = warp_figure(_t, _s_hat).to_html(full_html=False, include_plotlyjs=False) if _s_hat is not None else ""
    p4 = modes_figure(*_modes).to_html(full_html=False, include_plotlyjs=False) if _modes is not None else ""
    p5 = argand_figure(*_modes).to_html(full_html=False, include_plotlyjs=False) if _modes is not None else ""
    p6 = function_figure(_points, _f, _f_hat, _f_reg, _f_stats).to_html(full_html=False, include_plotlyjs=False) if _f is not None else ""
    return (HTML.replace("<!--PLOT1-->", p1).replace("<!--PLOT2-->", p2)
            .replace("<!--PLOT3-->", p3).replace("<!--PLOT4-->", p4)
            .replace("<!--PLOT5-->", p5).replace("<!--PLOT6-->", p6))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = page().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(self.address_string(), fmt % args)


def serve(host=HOST, port=PORT):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"http://{host}:{port}", flush=True)
    httpd.serve_forever()
