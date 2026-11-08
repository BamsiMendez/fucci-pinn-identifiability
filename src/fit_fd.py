"""Estimación de máxima verosimilitud con el solver de diferencias finitas
(referencia para validar la PINN). Uso:
    python fit_fd.py [--scenario 1|2] [--data archivo.xlsx]
"""
import argparse
import json
import numpy as np
from scipy.optimize import least_squares
from common import load_data, data_arrays, FDSolver, SIGMA, BOUNDS


def fit_mle(xs, R, G, scenario=1, x0=None, ic="cos"):
    fd = FDSolver(xs, R[0], G[0], ic=ic)
    names = ["D", "kr", "kg"] if scenario == 1 else ["Dr", "Dg", "kr", "kg"]
    lb = [BOUNDS[n][0] for n in names]
    ub = [BOUNDS[n][1] for n in names]
    scale = np.array([1000.0 if n.startswith("D") else 0.05 for n in names])
    if x0 is None:
        x0 = scale.copy()

    def unpack(p):
        p = p * scale
        return (p[0], p[0], p[1], p[2]) if scenario == 1 else tuple(p)

    def resid(p):
        ro, go = fd.at_obs(*unpack(p))
        # se excluye t=0 (es la condición inicial impuesta)
        return np.concatenate([(ro[1:] - R[1:]).ravel(), (go[1:] - G[1:]).ravel()]) / SIGMA

    res = least_squares(resid, np.asarray(x0) / scale, bounds=(np.array(lb) / scale, np.array(ub) / scale),
                        method="trf", x_scale=1.0, xtol=1e-10, ftol=1e-10)
    if not res.success or not np.isfinite(res.fun).all():
        raise RuntimeError("Falló el ajuste: " + res.message)
    est = dict(zip(names, (res.x * scale).tolist()))
    # Errores estándar aproximados (Fisher observada vía Jacobiano)
    J = res.jac / scale
    try:
        cov = np.linalg.inv(J.T @ J)
        se = dict(zip(names, np.sqrt(np.diag(cov)).tolist()))
    except np.linalg.LinAlgError:
        se = None
    return est, se, float(np.sum(res.fun ** 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, choices=[1, 2], default=1)
    ap.add_argument("--data", default=None)
    ap.add_argument("--ic", default="cos", choices=["cos", "interp"])
    a = ap.parse_args()
    df = load_data(a.data) if a.data else load_data()
    xs, ts, R, G = data_arrays(df)
    est, se, chi2 = fit_mle(xs, R, G, a.scenario, ic=a.ic)
    print(json.dumps({"mle": est, "se": se, "chi2": chi2}, indent=2))
