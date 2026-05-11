"""
readme_plots.py
---------------
Generates all required README comparison plots:
 
1. Contour + feasible region + optimization paths (simple1, simple2) x 2 algorithms = 4 plots
2. Objective vs iteration + max constraint violation vs iteration (simple2 only) x 2 algorithms = 4 plots
 
Run from AA222Project2 directory:
    python readme_plots.py
 
Saves all 8 plots as PNG files in the current directory.
"""
 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from project2_py.project2 import augmented_lagrangian, quadratic_penalty
 
# ── problem definitions (mirrors helpers.py, but with nolimit) ────────────────
 
def f1(x): return -x[0]*x[1] + 2/(3*np.sqrt(3))
def g1(x): return np.array([-x[1], -x[0]])
def c1(x): return np.array([x[0] + x[1]**2 - 1, -x[0] - x[1]])
 
def f2(x): return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
def g2(x): return np.array([2*(-1 + x[0] + 200*x[0]**3 - 200*x[0]*x[1]),
                             200*(-x[0]**2 + x[1])])
def c2(x): return np.array([(x[0]-1)**3 - x[1] + 1, x[0] + x[1] - 2])
 
PROBLEMS = {
    "simple1": (f1, g1, c1),
    "simple2": (f2, g2, c2),
}
 
# ── history-tracking wrappers ─────────────────────────────────────────────────
 
def make_tracked(f, g, c):
    """Returns (f_t, g_t, c_t, count, get_history).
    get_history() returns list of x visited during optimization."""
    ctr = [0]
    history = []
 
    def f_t(x):
        ctr[0] += 1
        history.append(x.copy())
        return f(x)
 
    def g_t(x):
        ctr[0] += 2
        return g(x)
 
    def c_t(x):
        ctr[0] += 1
        return c(x)
 
    def count():
        return ctr[0]
 
    def get_history():
        return history
 
    return f_t, g_t, c_t, count, get_history
 
 
# ── run algorithm and collect path ────────────────────────────────────────────
 
def run_with_history(algo_fn, algo_kwargs, f, g, c, x0):
    """Run algorithm with nolimit (n=np.inf) and return path."""
    f_t, g_t, c_t, count, get_history = make_tracked(f, g, c)
    algo_fn(f_t, g_t, c_t, x0, np.inf, count, **algo_kwargs)
    return np.array(get_history())
 
 
# ── plotting helpers ──────────────────────────────────────────────────────────
 
SEEDS = [0, 7, 42]  # three initial conditions
 
ALGO_CONFIGS = {
    "Augmented Lagrangian": {
        "fn": augmented_lagrangian,
        "simple1_kwargs": {"rho": 1.0, "gamma": 2.0, "refresh_jacobian": True},
        "simple2_kwargs": {"rho": 1.0, "gamma": 2.0, "refresh_jacobian": False},
        "color": "#e63946",
    },
    "Quadratic Penalty": {
        "fn": quadratic_penalty,
        "simple1_kwargs": {"rho": 1.0, "gamma": 3.0},
        "simple2_kwargs": {"rho": 1.0, "gamma": 3.0},
        "color": "#2a9d8f",
    },
}
 
AXIS = (-3, 3)
 
 
def get_x0(prob, seed):
    np.random.seed(seed)
    if prob == "simple1":
        return np.random.rand(2) * 2.0
    else:
        return np.random.rand(2) * 2.0 - 1.0
 
 
def feasible_mask(c_fn, X, Y):
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            cv = c_fn(np.array([X[i, j], Y[i, j]]))
            Z[i, j] = 1.0 if np.all(cv <= 0) else 0.0
    return Z
 
 
# ── Plot 1 & 2: contour + feasible region + paths ────────────────────────────
 
def plot_contour_paths(prob_name, algo_name, algo_cfg):
    f, g, c = PROBLEMS[prob_name]
    kwargs = algo_cfg[f"{prob_name}_kwargs"]
 
    x_lin = np.linspace(*AXIS, 300)
    X, Y = np.meshgrid(x_lin, x_lin)
 
    # Objective contour
    Z = np.vectorize(lambda x1, x2: f(np.array([x1, x2])))(X, Y)
    Z = np.clip(Z, np.percentile(Z, 2), np.percentile(Z, 98))
 
    # Feasible region
    F = feasible_mask(c, X, Y)
 
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.contourf(X, Y, Z, levels=30, cmap="YlOrRd", alpha=0.7)
    ax.contour(X, Y, Z, levels=15, colors="white", linewidths=0.4, alpha=0.5)
    ax.contourf(X, Y, F, levels=[0.5, 1.5], colors=["#a8dadc"], alpha=0.45)
    ax.contour(X, Y, F, levels=[0.5], colors=["#1d3557"], linewidths=1.5)
 
    # Optimal point
    if prob_name == "simple1":
        xs, ys = 2/3, 1/np.sqrt(3)
    else:
        xs, ys = 1.0, 1.0
    ax.plot(xs, ys, "*", color="gold", markersize=14, zorder=10, label="Optimum")
 
    # Paths for each seed
    path_colors = ["#ffffff", "#f4d35e", "#ee964b"]
    for i, seed in enumerate(SEEDS):
        x0 = get_x0(prob_name, seed)
        path = run_with_history(algo_cfg["fn"], kwargs, f, g, c, x0)
        # Clip path to axis limits for display
        path = np.clip(path, AXIS[0], AXIS[1])
        ax.plot(path[:, 0], path[:, 1], "-o",
                color=path_colors[i], markersize=2, linewidth=1.2,
                alpha=0.85, label=f"Seed {seed}")
        ax.plot(path[0, 0], path[0, 1], "s",
                color=path_colors[i], markersize=6, zorder=9)
 
    ax.set_xlim(*AXIS)
    ax.set_ylim(*AXIS)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title(f"{prob_name} — {algo_name}\nContour + Feasible Region + Paths", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
 
    fname = f"plot_{prob_name}_{algo_name.lower().replace(' ', '_')}_contour.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")
    return fname
 
 
# ── Plot 3 & 4: objective + constraint violation vs iteration (simple2 only) ──
 
def run_convergence(algo_fn, algo_kwargs, f, g, c, x0):
    """Returns (f_history, max_cv_history) per f-evaluation step."""
    f_hist = []
    cv_hist = []
    ctr = [0]
 
    def f_t(x):
        ctr[0] += 1
        fx = f(x)
        cv = np.max(np.maximum(0.0, c(x)))
        f_hist.append(fx)
        cv_hist.append(cv)
        return fx
 
    def g_t(x):
        ctr[0] += 2
        return g(x)
 
    def c_t(x):
        ctr[0] += 1
        return c(x)
 
    def count():
        return ctr[0]
 
    algo_fn(f_t, g_t, c_t, x0, np.inf, count, **algo_kwargs)
    return np.array(f_hist), np.array(cv_hist)
 
 
def plot_convergence(algo_name, algo_cfg):
    f, g, c = PROBLEMS["simple2"]
    kwargs = algo_cfg["simple2_kwargs"]
    color = algo_cfg["color"]
 
    seed_colors = ["#e63946", "#457b9d", "#2a9d8f"]
 
    fig_f, ax_f = plt.subplots(figsize=(6, 4))
    fig_cv, ax_cv = plt.subplots(figsize=(6, 4))
 
    for i, seed in enumerate(SEEDS):
        x0 = get_x0("simple2", seed)
        f_hist, cv_hist = run_convergence(algo_cfg["fn"], kwargs, f, g, c, x0)
 
        iters = np.arange(len(f_hist))
        ax_f.plot(iters, f_hist, color=seed_colors[i], linewidth=1.5,
                  alpha=0.85, label=f"Seed {seed}")
        ax_cv.plot(iters, cv_hist, color=seed_colors[i], linewidth=1.5,
                   alpha=0.85, label=f"Seed {seed}")
 
    ax_f.set_xlabel("Iteration (f-eval count)", fontsize=11)
    ax_f.set_ylabel("Objective f(x)", fontsize=11)
    ax_f.set_title(f"simple2 — {algo_name}\nObjective vs Iteration", fontsize=11)
    ax_f.set_yscale("symlog", linthresh=1e-3)
    ax_f.legend(fontsize=9)
    ax_f.grid(True, alpha=0.3)
    fig_f.tight_layout()
 
    ax_cv.set_xlabel("Iteration (f-eval count)", fontsize=11)
    ax_cv.set_ylabel("Max Constraint Violation", fontsize=11)
    ax_cv.set_title(f"simple2 — {algo_name}\nMax Constraint Violation vs Iteration", fontsize=11)
    ax_cv.set_yscale("symlog", linthresh=1e-6)
    ax_cv.legend(fontsize=9)
    ax_cv.grid(True, alpha=0.3)
    fig_cv.tight_layout()
 
    fname_f  = f"plot_simple2_{algo_name.lower().replace(' ', '_')}_objective.png"
    fname_cv = f"plot_simple2_{algo_name.lower().replace(' ', '_')}_violation.png"
    fig_f.savefig(fname_f, dpi=150);  plt.close(fig_f);  print(f"Saved {fname_f}")
    fig_cv.savefig(fname_cv, dpi=150); plt.close(fig_cv); print(f"Saved {fname_cv}")
    return fname_f, fname_cv
 
 
# ── main ──────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    saved = []
 
    print("Generating contour plots...")
    for prob in ["simple1", "simple2"]:
        for algo_name, algo_cfg in ALGO_CONFIGS.items():
            fname = plot_contour_paths(prob, algo_name, algo_cfg)
            saved.append(fname)
 
    print("\nGenerating convergence plots (simple2 only)...")
    for algo_name, algo_cfg in ALGO_CONFIGS.items():
        fnames = plot_convergence(algo_name, algo_cfg)
        saved.extend(fnames)
 
    print(f"\nDone! {len(saved)} plots saved:")
    for f in saved:
        print(f"  {f}")