#
# File: project2.py
#

## top-level submission file

import numpy as np

# base optimize function
def optimize(f, g, c, x0, n, count, prob):
    
    if prob in ("simple1", "simple2", "simple3"):
        return augmented_lagrangian(f, g, c, x0, n, count)
    else:
        # Secret problems: higher-dimensional, more evaluations available.
        # AL is robust; use it with a longer inner loop.
        return augmented_lagrangian(f, g, c, x0, n, count, inner_iters=200)
    
# method 1  
def augmented_lagrangian(f, g, c, x0, n, count,
                         rho=1.0, gamma=5.0, inner_iters=100):
    x = x0.copy().astype(float)
    lam = np.zeros(len(c(x)))   # Lagrange multipliers (one per constraint)
 
    x_best = x.copy()
    f_best = np.inf
    feasible_best = False
 
    while count() < n:
        # --- Define augmented Lagrangian and its gradient ---
        def aug_obj(x):
            cx = c(x)
            shifted = lam / rho + cx
            penalty = (rho / 2.0) * np.sum(np.maximum(0.0, shifted) ** 2)
            return f(x) + penalty
 
        def aug_grad(x):
            cx = c(x)
            shifted = lam / rho + cx
            active = (shifted > 0).astype(float)
            # gradient of penalty w.r.t. x: rho * max(0, shifted) * grad_c_i(x)
            # We approximate grad_c via finite differences (as required by rules)
            gx = g(x)  # gradient of f (costs 2 evals)
            grad_penalty = finite_diff_constraint_grad(c, x, cx, active, count, n)
            return gx + grad_penalty
 
        # --- Inner loop: gradient descent with backtracking line search ---
        x = gradient_descent(aug_obj, aug_grad, x, inner_iters, count, n)
 
        # --- Track best feasible point ---
        cx = c(x)
        is_feasible = np.all(cx <= 0)
        fx = f(x)
 
        if is_feasible:
            if not feasible_best or fx < f_best:
                x_best = x.copy()
                f_best = fx
                feasible_best = True
        elif not feasible_best:
            # No feasible point yet; track least-infeasible
            viol = np.sum(np.maximum(0, cx))
            if viol < f_best:
                x_best = x.copy()
                f_best = viol
 
        # --- Update multipliers and penalty ---
        lam = np.maximum(0.0, lam + rho * cx)
        rho = min(rho * gamma, 1e8)
 
        if count() >= n:
            break
 
    return x_best

# helpers 
def finite_diff_constraint_grad(c, x, cx, active_mask, count, n, eps=1e-5):
    """
    Finite difference gradient of sum_i active_i * rho * max(0, shifted_i) * c_i(x).
    Since rho and lam are captured in active_mask * shifted already, we just need:
        d/dx [ sum_i active_i * c_i(x) ]
    which is the Jacobian of c at x, weighted by active_mask.
    We use forward differences column by column.
    """
    if not np.any(active_mask):
        return np.zeros_like(x)
 
    grad = np.zeros_like(x)
    for j in range(len(x)):
        if count() + 1 > n:
            break
        xp = x.copy()
        xp[j] += eps
        cxp = c(xp)
        # weighted sum of constraint changes in active directions
        grad[j] = np.dot(active_mask, (cxp - cx)) / eps
    return grad
 
def gradient_descent(obj, grad, x, max_iters, count, n,
                     alpha=1.0, beta=0.5, c_armijo=1e-4):
    """
    Gradient descent with backtracking line search (Armijo condition).
    Stops when budget is exhausted or gradient is near zero.
    """
    x = x.copy()
    for _ in range(max_iters):
        if count() >= n:
            break
        gx = grad(x)
        if np.linalg.norm(gx) < 1e-8:
            break
 
        # Backtracking line search
        fx = obj(x)
        step = alpha
        for _ in range(30):
            if count() >= n:
                break
            x_new = x - step * gx
            if obj(x_new) <= fx - c_armijo * step * np.dot(gx, gx):
                break
            step *= beta
        else:
            x_new = x - step * gx
 
        x = x_new
 
    return x

# method 2

def quadratic_penalty(f, g, c, x0, n, count,
                      rho=1.0, gamma=10.0, inner_iters=100):
    """
    Quadratic penalty method for inequality constraints c(x) <= 0.
 
    Penalized objective:
        f_pen(x) = f(x) + rho * sum_i max(0, c_i(x))^2
 
    Increases rho geometrically each outer iteration.
    Inner minimization uses gradient descent with backtracking line search.
    """
    x = x0.copy().astype(float)
    x_best = x.copy()
    f_best = np.inf
    feasible_best = False
 
    while count() < n:
        def pen_obj(x):
            cx = c(x)
            return f(x) + rho * np.sum(np.maximum(0.0, cx) ** 2)
 
        def pen_grad(x):
            cx = c(x)
            gx = g(x)
            active = (cx > 0).astype(float)
            grad_penalty = finite_diff_constraint_grad(c, x, cx, 2 * rho * np.maximum(0, cx), count, n)
            return gx + grad_penalty
 
        x = gradient_descent(pen_obj, pen_grad, x, inner_iters, count, n)
 
        cx = c(x)
        is_feasible = np.all(cx <= 0)
        fx = f(x)
 
        if is_feasible:
            if not feasible_best or fx < f_best:
                x_best = x.copy()
                f_best = fx
                feasible_best = True
        elif not feasible_best:
            viol = np.sum(np.maximum(0, cx))
            if viol < f_best:
                x_best = x.copy()
                f_best = viol
 
        rho = min(rho * gamma, 1e8)
 
        if count() >= n:
            break
 
    return x_best
