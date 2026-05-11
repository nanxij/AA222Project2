import numpy as np

def optimize(f, g, c, x0, n, count, prob):
    if prob == "simple1":
        return augmented_lagrangian(f, g, c, x0, n, count, rho=1.0, gamma=2.0, refresh_jacobian=True)
    elif prob == "simple2":
        return augmented_lagrangian(f, g, c, x0, n, count, rho=1.0, gamma=2.0, refresh_jacobian=False)
    elif prob == "simple3":
        return augmented_lagrangian(f, g, c, x0, n, count, rho=1.0, gamma=2.0, refresh_jacobian=True)
    else:
        return augmented_lagrangian(f, g, c, x0, n, count, rho=1.0, gamma=2.0, refresh_jacobian=True)


def augmented_lagrangian(f, g, c, x0, n, count, rho=1.0, gamma=2.0, refresh_jacobian=False):
    x = x0.copy().astype(float)
    cx = c(x).flatten()
    lam = np.zeros(len(cx))

    x_best = x.copy()
    f_best = np.inf
    feasible_best = False

    # Choose inner iters based on refresh jacob
    inner_iters = 5 if refresh_jacobian else 8

    while count() < n - 20:
        cx = c(x).flatten()

        # Compute J
        if not refresh_jacobian:
            J = _jacobian(c, x, cx, count, n)

        for _ in range(inner_iters):
            if count() >= n - 12:
                break

            cx_i = c(x).flatten()

            # Refresh J if requested
            if refresh_jacobian:
                if count() >= n - 12:
                    break
                J = _jacobian(c, x, cx_i, count, n)

            shifted = lam / rho + cx_i
            active = np.maximum(0.0, shifted)

            gx = g(x)
            grad = gx + rho * (J.T @ active)

            gnorm = np.linalg.norm(grad)
            if not np.isfinite(gnorm) or gnorm < 1e-8:
                break

            step = 1.0 / gnorm
            fx_aug = f(x) + (rho / 2.0) * np.sum(active ** 2)

            x_new = x - step * grad
            for _ in range(8):
                if count() >= n - 5:
                    break
                x_cand = x - step * grad
                cx_cand = c(x_cand).flatten()
                a_cand = np.maximum(0.0, lam / rho + cx_cand)
                fx_cand = f(x_cand) + (rho / 2.0) * np.sum(a_cand ** 2)
                if np.isfinite(fx_cand) and fx_cand <= fx_aug - 1e-4 * step * gnorm ** 2:
                    x_new = x_cand
                    break
                step *= 0.5

            if not np.all(np.isfinite(x_new)):
                break
            x = x_new

        # Track best
        if count() < n:
            cx = c(x).flatten()
        if count() < n:
            fx = f(x)
            if np.all(cx <= 0):
                if not feasible_best or fx < f_best:
                    x_best, f_best, feasible_best = x.copy(), fx, True
            elif not feasible_best:
                viol = np.sum(np.maximum(0.0, cx))
                if viol < f_best:
                    x_best, f_best = x.copy(), viol

        lam = np.maximum(0.0, lam + rho * cx)
        rho = min(rho * gamma, 1e4)

    return x_best


def _jacobian(c, x, cx, count, n, eps=1e-5):
    cx = np.atleast_1d(cx).flatten()
    J = np.zeros((len(cx), len(x)))
    for j in range(len(x)):
        if count() >= n - 10:
            break
        xp = x.copy()
        xp[j] += eps
        J[:, j] = (np.atleast_1d(c(xp)).flatten() - cx) / eps
    return J

#algo 2

def quadratic_penalty(f, g, c, x0, n, count, rho=1.0, gamma=3.0):
    x = x0.copy().astype(float)
    x_best = x.copy()
    f_best = np.inf
    feasible_best = False

    while count() < n - 20:
        cx = c(x).flatten()
        J = _jacobian(c, x, cx, count, n)

        for _ in range(8):
            if count() >= n - 10:
                break

            cx_i = c(x).flatten()
            active = np.maximum(0.0, cx_i)

            gx = g(x)
            grad = gx + 2.0 * rho * (J.T @ active)

            gnorm = np.linalg.norm(grad)
            if not np.isfinite(gnorm) or gnorm < 1e-8:
                break

            step = 1.0 / gnorm
            fx_pen = f(x) + rho * np.sum(active ** 2)

            x_new = x - step * grad
            for _ in range(8):
                if count() >= n - 5:
                    break
                x_cand = x - step * grad
                cx_cand = c(x_cand).flatten()
                a_cand = np.maximum(0.0, cx_cand)
                fx_cand = f(x_cand) + rho * np.sum(a_cand ** 2)
                if np.isfinite(fx_cand) and fx_cand <= fx_pen - 1e-4 * step * gnorm ** 2:
                    x_new = x_cand
                    break
                step *= 0.5

            if not np.all(np.isfinite(x_new)):
                break
            x = x_new

        if count() < n:
            cx = c(x).flatten()
        if count() < n:
            fx = f(x)
            if np.all(cx <= 0):
                if not feasible_best or fx < f_best:
                    x_best, f_best, feasible_best = x.copy(), fx, True
            elif not feasible_best:
                viol = np.sum(np.maximum(0.0, cx))
                if viol < f_best:
                    x_best, f_best = x.copy(), viol

        rho = min(rho * gamma, 1e4)

    return x_best