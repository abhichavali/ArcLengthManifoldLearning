"""Web visualizer for the unordered 1D manifold. Run: make laplace-unordered"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HOST, PORT = "127.0.0.1", 8001
_points = None
_ordered = None
_bench = None

HTML = """<!doctype html>
<meta charset="utf-8">
<title>Unordered 1D manifold</title>
<style>
  body { font: 16px/1.4 system-ui, sans-serif; margin: 0; }
  nav { display: flex; gap: .5rem; padding: .75rem 1rem; border-bottom: 1px solid #ccc; }
  nav button { padding: .4rem .8rem; cursor: pointer; }
  nav button.active { font-weight: 700; }
  .panel { display: none; }
  .panel.active { display: block; }
</style>
<nav>
  <button class="active" data-tab="1">1. Unordered samples</button>
  <button data-tab="2">2. Recovered order</button>
  <button data-tab="3">3. vs diffusion maps</button>
</nav>
<section class="panel active" data-tab="1"><!--PLOT1--></section>
<section class="panel" data-tab="2"><!--PLOT2--></section>
<section class="panel" data-tab="3"><!--PLOT3--></section>
<script>
document.querySelector("nav").onclick = e => {
  const t = e.target.dataset.tab;
  if (!t) return;
  document.querySelectorAll("[data-tab]").forEach(el => el.classList.toggle("active", el.dataset.tab === t));
  document.querySelectorAll(".plotly-graph-div").forEach(gd => Plotly.Plots.resize(gd));
};
</script>
"""


def show_samples(points):
    global _points
    _points = np.asarray(points)


def show_ordered(points, rank, angle):
    global _ordered
    _ordered = (np.asarray(points), np.asarray(rank), np.asarray(angle))


def show_benchmark(rows):
    global _bench
    _bench = np.asarray(rows, dtype=float)


def benchmark_figure(rows):
    n, t_order, t_arc, t_dm, a_rmse, d_rmse = rows.T
    series = [("order recovery (Lanczos k=3)", t_order, "#ff7f0e"), ("arc-length fit given order", t_arc, "#1f77b4"),
              ("order + arc-length", t_order + t_arc, "#2ca02c"), ("full diffusion map fit", t_dm, "#d62728")]
    fig = make_subplots(rows=2, cols=2, row_heights=[0.65, 0.35],
                        specs=[[{}, {}], [{"type": "table", "colspan": 2}, None]],
                        subplot_titles=["time to approximate T on an unordered cloud (median, log-log)",
                                        "RMSE vs N (33 coefficients each)", ""])
    for name, y, col in series:
        fig.add_trace(go.Scatter(x=n, y=y, mode="lines+markers", name=name, line=dict(color=col)), row=1, col=1)
    fig.add_trace(go.Scatter(x=n, y=a_rmse, mode="lines+markers", name="arc-length", line=dict(color="#1f77b4"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=n, y=d_rmse, mode="lines+markers", name="diffusion", line=dict(color="#d62728"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Table(
        header=dict(values=["N", "order ms", "arc-length ms", "order + arc ms", "diffusion ms", "speedup", "arc RMSE", "diffusion RMSE"]),
        cells=dict(values=[n.astype(int), np.round(t_order, 3), np.round(t_arc, 3), np.round(t_order + t_arc, 3),
                           np.round(t_dm, 3), np.round(t_dm / (t_order + t_arc), 1), np.round(a_rmse, 4), np.round(d_rmse, 4)]),
    ), row=2, col=1)
    fig.update_xaxes(type="log", title_text="N", row=1, col=1)
    fig.update_yaxes(type="log", title_text="ms", row=1, col=1)
    fig.update_xaxes(type="log", title_text="N", row=1, col=2)
    fig.update_yaxes(title_text="RMSE", row=1, col=2)
    fig.update_layout(title="unordered cloud: recover order via 3-vector Lanczos, then arc-length FFT, vs full dense eigensolve",
                      height=800, margin=dict(l=40, r=20, t=80, b=20))
    return fig


def _labelled_scatter(points, labels, title, cbar, hover=None):
    fig = go.Figure(go.Scatter(
        x=points[:, 0], y=points[:, 1], mode="markers+text", text=labels.astype(str),
        textposition="top center", textfont=dict(size=9), customdata=hover,
        marker=dict(size=6, color=labels, colorscale="Viridis", colorbar=dict(title=cbar)),
        hovertemplate="#%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}" + ("<br>angle=%{customdata:.3f}" if hover is not None else "") + "<extra></extra>",
    ))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="y", margin=dict(l=40, r=20, t=40, b=40), height=700)
    return fig


def samples_figure(points):
    return _labelled_scatter(points, np.arange(len(points)),
                             f"{len(points)} unordered samples of γ(θ) in R², labelled by draw order", "sample #")


def ordered_figure(points, rank, angle):
    return _labelled_scatter(points, rank,
                             "same points relabelled by atan2(λ₂ψ₂, λ₁ψ₁) rank from a 3-vector Lanczos diffusion map",
                             "order", hover=angle)


def page():
    p1 = samples_figure(_points).to_html(full_html=False, include_plotlyjs="cdn")
    p2 = ordered_figure(*_ordered).to_html(full_html=False, include_plotlyjs=False) if _ordered is not None else ""
    p3 = benchmark_figure(_bench).to_html(full_html=False, include_plotlyjs=False) if _bench is not None else ""
    return HTML.replace("<!--PLOT1-->", p1).replace("<!--PLOT2-->", p2).replace("<!--PLOT3-->", p3)


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
