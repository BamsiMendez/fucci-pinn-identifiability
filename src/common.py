"""Utilidades comunes: carga de datos y solver de referencia (diferencias finitas)
para el modelo FUCCI de Simpson et al. (2020), ecs. (3.1)-(3.2):

    r_t = Dr r_xx - kr r + 2 kg g (1 - r - g)
    g_t = Dg g_xx + kr r -   kg g (1 - r - g)

con flujo nulo en x = 0 y x = L, y condición inicial dada por los datos en t = 0.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

DATA_PATH = Path(__file__).resolve().parents[1] / "data/datos_reorganizados.csv"
SIGMA = 0.05          # desviación estándar del ruido (fijada en el artículo)
L = 1242.0            # micras
T = 48.0              # horas
TIMES = np.array([0.0, 16.0, 32.0, 48.0])

# Cotas biológicas (sección 4.1 del artículo)
BOUNDS = {"D": (1.0, 2000.0), "Dr": (1.0, 2000.0), "Dg": (1.0, 2000.0),
          "kr": (1e-4, 0.2), "kg": (1e-4, 0.2)}


def load_data(path=DATA_PATH):
    """Devuelve DataFrame con columnas x, t, r, g (densidades adimensionales)."""
    path = Path(path)
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    df.columns = ["x", "t", "count_r", "count_g", "r", "g"]
    return df[["x", "t", "r", "g"]].astype(float).sort_values(["t", "x"]).reset_index(drop=True)


def data_arrays(df):
    """xs (24,), ts (4,), R (4,24), G (4,24)."""
    xs = np.sort(df.x.unique())
    ts = np.sort(df.t.unique())
    R = df.pivot(index="t", columns="x", values="r").loc[ts, xs].values
    G = df.pivot(index="t", columns="x", values="g").loc[ts, xs].values
    return xs, ts, R, G


K_IC = 8   # nº de modos coseno para la condición inicial suavizada


def cosine_ic_coeffs(x_obs, y0, K=K_IC):
    """Ajuste por mínimos cuadrados de y0(x) ≈ Σ_{k=0..K} c_k cos(kπx/L).
    Cada modo cumple el flujo nulo en x=0 y x=L, así que la CI es compatible con la EDP."""
    A = np.cos(np.pi * np.outer(np.asarray(x_obs) / L, np.arange(K + 1)))
    c, *_ = np.linalg.lstsq(A, y0, rcond=None)
    return c


def cosine_eval(c, x):
    return np.cos(np.pi * np.outer(np.asarray(x) / L, np.arange(len(c)))) @ c


class FDSolver:
    """Método de líneas, diferencias centradas de 2º orden, Neumann homogéneo."""

    def __init__(self, x_obs, r0_obs, g0_obs, nx=139, ic="cos"):
        self.x = np.linspace(0.0, L, nx)
        self.dx = self.x[1] - self.x[0]
        self.x_obs = x_obs
        if ic == "interp":   # interpolación lineal de los datos en t = 0 (como en el artículo)
            self.r0 = np.interp(self.x, x_obs, r0_obs)
            self.g0 = np.interp(self.x, x_obs, g0_obs)
        else:                # serie coseno suavizada (misma CI que usa la PINN)
            self.r0 = cosine_eval(cosine_ic_coeffs(x_obs, r0_obs), self.x)
            self.g0 = cosine_eval(cosine_ic_coeffs(x_obs, g0_obs), self.x)
        n = nx
        # Laplaciano con nodos fantasma reflejados (flujo nulo)
        Lap = (np.diag(-2.0 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1))
        Lap[0, 1] = 2.0
        Lap[-1, -2] = 2.0
        self.Lap = Lap / self.dx ** 2
        self.n = n

    def solve(self, Dr, Dg, kr, kg, t_eval=TIMES):
        n, Lap = self.n, self.Lap

        def rhs(t, y):
            r, g = y[:n], y[n:]
            c = kg * g * (1.0 - r - g)
            return np.concatenate([Dr * (Lap @ r) - kr * r + 2.0 * c,
                                   Dg * (Lap @ g) + kr * r - c])

        def jac(t, y):
            r, g = y[:n], y[n:]
            s = 1.0 - r - g
            I = np.eye(n)
            dc_dr = np.diag(-kg * g)
            dc_dg = np.diag(kg * (s - g))
            J = np.block([[Dr * Lap - kr * I + 2 * dc_dr, 2 * dc_dg],
                          [kr * I - dc_dr, Dg * Lap - dc_dg]])
            return J

        sol = solve_ivp(rhs, (0.0, t_eval[-1]), np.concatenate([self.r0, self.g0]),
                        t_eval=t_eval, method="BDF", jac=jac, rtol=1e-7, atol=1e-9)
        if not sol.success or not np.isfinite(sol.y).all():
            raise RuntimeError("Falló el integrador: " + sol.message)
        r = sol.y[:n].T
        g = sol.y[n:].T
        return r, g  # (nt, nx)

    def at_obs(self, Dr, Dg, kr, kg, t_eval=TIMES):
        r, g = self.solve(Dr, Dg, kr, kg, t_eval)
        ro = np.array([np.interp(self.x_obs, self.x, ri) for ri in r])
        go = np.array([np.interp(self.x_obs, self.x, gi) for gi in g])
        return ro, go
