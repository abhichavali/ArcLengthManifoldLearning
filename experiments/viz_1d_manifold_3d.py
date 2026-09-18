# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "plotly"]
# ///
"""Interactive 3D view of a 1D manifold in R^3, a function on it, and its natural modes.

Run: uv run experiments/viz_1d_manifold_3d.py   -> experiments/viz_1d_manifold_3d.html
Dropdown switches what colors the curve; drag to rotate.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

rng = np.random.default_rng(0)
N, K = 600, 6

# --- closed space curve (trefoil), non-uniform in t ---
t = np.sort(rng.uniform(0, 2 * np.pi, N))
X = np.c_[np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t), -np.sin(3 * t)]

# --- arc length (closed) ---
seg = np.linalg.norm(np.diff(X, axis=0, append=X[:1]), axis=1)
s = np.r_[0, np.cumsum(seg[:-1])]
L = seg.sum()

def mode(k):  # Laplacian eigenfunctions on a closed curve: Fourier modes in arc length
    m = (k + 1) // 2
    return np.cos(2 * np.pi * m * s / L) if k % 2 else np.sin(2 * np.pi * m * s / L)

f = np.sin(2 * np.pi * 2 * s / L) + 0.5 * np.cos(2 * np.pi * 5 * s / L) + 0.1 * rng.standard_normal(N)

# --- graph Laplacian comparison ---
D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
W = np.exp(-D2 / (2 * seg.max()) ** 2)
_, V = np.linalg.eigh(np.diag(W.sum(1)) - W)

fields = {"parameter t": t, "arc length s": s, "function f": f}
fields |= {f"arc-length mode {k}": mode(k) for k in range(K)}
fields |= {f"graph Laplacian eigvec {k}": V[:, k] for k in range(K)}

fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scene", "rowspan": 2}, {"type": "xy"}], [None, {"type": "xy"}]],
                    column_widths=[0.5, 0.5], subplot_titles=["manifold in R³", "value vs arc length s", "value vs parameter t"])
fig.add_trace(go.Scatter3d(
    x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers",
    marker=dict(size=4, color=t, colorscale="RdBu_r", showscale=True),
    customdata=np.c_[s, f], hovertemplate="s=%{customdata[0]:.2f}<br>f=%{customdata[1]:.2f}<extra></extra>",
), row=1, col=1)
fig.add_trace(go.Scatter(x=s, y=t, mode="markers", marker=dict(size=4, color=t, colorscale="RdBu_r"),
                         hovertemplate="s=%{x:.2f}<br>%{y:.3f}<extra></extra>"), row=1, col=2)
fig.add_trace(go.Scatter(x=t, y=t, mode="markers", marker=dict(size=4, color=t, colorscale="RdBu_r"),
                         hovertemplate="t=%{x:.2f}<br>%{y:.3f}<extra></extra>"), row=2, col=2)
fig.update_xaxes(title_text="arc length s", row=1, col=2)
fig.update_xaxes(title_text="parameter t", row=2, col=2)
sym = lambda c: {"cmin": -abs(c).max(), "cmax": abs(c).max()}  # symmetric scale so constant modes read as constant
fig.update_layout(
    title="color: parameter t", scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=60, b=0), showlegend=False,
    updatemenus=[dict(x=0, y=1, buttons=[
        dict(label=name, method="update",
             args=[{"marker.color": [c] * 3, "y": [X[:, 1], c, c],  # restyle all traces: 3D color + both 1D graphs
                    **({f"marker.{k}": [v] * 3 for k, v in sym(c).items()} if "mode" in name or "eigvec" in name else {"marker.cmin": [None] * 3, "marker.cmax": [None] * 3})},
                   {"title": f"color: {name}"}])
        for name, c in fields.items()])],
)
out = __file__.replace(".py", ".html")
fig.write_html(out, include_plotlyjs="cdn")
print("wrote", out)
