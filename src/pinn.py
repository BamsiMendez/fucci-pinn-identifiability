"""PINN para el modelo FUCCI de reacción-difusión (Simpson et al., 2020).

Red  N(x, t) -> (r, g)  con parámetros físicos entrenables  θ = (D, kr, kg)  [escenario 1]
o  θ = (Dr, Dg, kr, kg)  [escenario 2].

Variables adimensionales: ξ = x/L, τ = t/T  (L = 1242 μm, T = 48 h). La EDP queda
    r_τ = (Dr T/L²) r_ξξ - T kr r + 2 T kg g (1 - r - g)
    g_τ = (Dg T/L²) g_ξξ + T kr r -   T kg g (1 - r - g)

Diseño (elegido para que la inferencia de θ sea fiable):
  * Flujo nulo EXACTO: la red recibe cos(kπξ), k=1..3, cuyas derivadas se anulan en ξ=0,1.
  * CI EXACTA: r(ξ,τ) = r0(ξ) + τ·N_r(ξ,τ), con r0 = serie coseno (K=8) ajustada a los datos
    en t=0 (la misma CI usa el solver de referencia de diferencias finitas).
  * Pérdida = L_data/σ² + w_pde·L_pde, con continuación del peso w_pde (de pequeño a grande):
    al principio la red sigue los datos; al final es una solución precisa de la EDP.
  * Adam (red + log θ) y refinamiento L-BFGS; colocación remuestreada para evitar sobreajuste
    del residuo.

Uso:
    python pinn.py --scenario 1 --seed 0 --out results/real_s1_seed0.json
    python pinn.py --data synth.xlsx --scenario 2 --seed 3
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn

from common import load_data, data_arrays, cosine_ic_coeffs, SIGMA, L, T

torch.set_default_dtype(torch.float64)

PARAM_SCALE = {"D": 1000.0, "Dr": 1000.0, "Dg": 1000.0, "kr": 0.05, "kg": 0.05}
OUT_SCALE = 0.1


class FucciPINN(nn.Module):
    def __init__(self, ic_r, ic_g, scenario=1, init=None, width=32, depth=4):
        super().__init__()
        self.scenario = scenario
        self.names = ["D", "kr", "kg"] if scenario == 1 else ["Dr", "Dg", "kr", "kg"]
        layers, d_in = [], 4
        for _ in range(depth):
            lin = nn.Linear(d_in, width)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            d_in = width
        out = nn.Linear(d_in, 2)
        nn.init.xavier_normal_(out.weight, gain=0.1)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.body = nn.Sequential(*layers)
        # coeficientes coseno de la CI (fijos)
        self.register_buffer("c_r", torch.tensor(ic_r))
        self.register_buffer("c_g", torch.tensor(ic_g))
        self.register_buffer("k_ic", torch.arange(len(ic_r), dtype=torch.float64))
        init = init or {}
        self.log_p = nn.Parameter(torch.tensor(
            [math.log(init.get(n, PARAM_SCALE[n]) / PARAM_SCALE[n]) for n in self.names]))
        self.register_buffer("p_scale", torch.tensor([PARAM_SCALE[n] for n in self.names]))

    # ---- campo (r, g) ----
    def forward(self, xi, tau):
        a = math.pi * xi
        z = torch.cat([torch.cos(a), torch.cos(2 * a), torch.cos(3 * a), 2 * tau - 1], dim=1)
        o = OUT_SCALE * self.body(z)
        basis = torch.cos(a * self.k_ic)            # (n, K+1)
        r0 = basis @ self.c_r.unsqueeze(1)
        g0 = basis @ self.c_g.unsqueeze(1)
        return r0 + tau * o[:, :1], g0 + tau * o[:, 1:]

    # ---- parámetros físicos ----
    def params(self):
        p = dict(zip(self.names, torch.exp(self.log_p) * self.p_scale))
        if self.scenario == 1:
            p["Dr"] = p["Dg"] = p["D"]
        return p

    def params_float(self):
        with torch.no_grad():
            v = (torch.exp(self.log_p) * self.p_scale).tolist()
        return dict(zip(self.names, v))

    # ---- residuo de la EDP ----
    def residual(self, xi, tau):
        xi = xi.clone().requires_grad_(True)
        tau = tau.clone().requires_grad_(True)
        r, g = self(xi, tau)
        ones = torch.ones_like(r)
        r_t, r_x = torch.autograd.grad(r, (tau, xi), ones, create_graph=True)
        g_t, g_x = torch.autograd.grad(g, (tau, xi), ones, create_graph=True)
        r_xx = torch.autograd.grad(r_x, xi, ones, create_graph=True)[0]
        g_xx = torch.autograd.grad(g_x, xi, ones, create_graph=True)[0]
        p = self.params()
        c = T * p["kg"] * g * (1 - r - g)
        fr = r_t - (p["Dr"] * T / L ** 2) * r_xx + T * p["kr"] * r - 2 * c
        fg = g_t - (p["Dg"] * T / L ** 2) * g_xx - T * p["kr"] * r + c
        return fr, fg


def sample_colloc(n, gen):
    xi = torch.rand(n, 1, generator=gen)
    tau = torch.rand(n, 1, generator=gen)
    m = n // 5                      # refuerzo cerca de t = 0
    tau[:m] = tau[:m] * 0.15
    return xi, tau


def build_model(df, scenario=1, init=None, width=32, depth=4):
    xs, ts, R, G = data_arrays(df)
    return FucciPINN(cosine_ic_coeffs(xs, R[0]), cosine_ic_coeffs(xs, G[0]),
                     scenario, init, width, depth)


def train(df, scenario=1, seed=0, init=None, n_adam=8000, n_lbfgs=2500, n_colloc=1500,
          w_pde_start=10.0, w_pde=2e3, lr=2e-3, lr_param=2e-2, width=32, depth=4,
          verbose=True, log_every=1000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed + 12345)
    xs, ts, R, G = data_arrays(df)

    mask = ts > 0
    Xd, Td = np.meshgrid(xs, ts[mask])
    xi_d = torch.tensor(Xd.ravel() / L).unsqueeze(1)
    tau_d = torch.tensor(Td.ravel() / T).unsqueeze(1)
    r_d = torch.tensor(R[mask].ravel()).unsqueeze(1)
    g_d = torch.tensor(G[mask].ravel()).unsqueeze(1)

    model = build_model(df, scenario, init, width, depth)

    def losses(xi_c, tau_c, w):
        r, g = model(xi_d, tau_d)
        l_data = ((r - r_d) ** 2 + (g - g_d) ** 2).mean() / SIGMA ** 2
        fr, fg = model.residual(xi_c, tau_c)
        l_pde = (fr ** 2 + fg ** 2).mean()
        return l_data + w * l_pde, l_data, l_pde

    def log(tag, it, w, ld, lp):
        if verbose:
            print(f"[{tag} {it:6d}] w_pde={w:.1e} data={ld:.4e} pde={lp:.3e} | " +
                  " ".join(f"{k}={v:.4g}" for k, v in model.params_float().items()), flush=True)

    history = []
    t0 = time.time()

    # ---------- Adam con continuación geométrica de w_pde ----------
    opt = torch.optim.Adam([{"params": model.body.parameters(), "lr": lr},
                            {"params": [model.log_p], "lr": lr_param}])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1 ** (1 / max(n_adam, 1)))
    ramp = int(0.6 * n_adam)
    for it in range(n_adam):
        if it % 100 == 0:
            xi_c, tau_c = sample_colloc(n_colloc, gen)
        w = w_pde_start * (w_pde / w_pde_start) ** min(1.0, it / max(ramp, 1))
        opt.zero_grad()
        loss, ld, lp = losses(xi_c, tau_c, w)
        loss.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == n_adam - 1:
            history.append(dict(phase="adam", it=it, w_pde=w, data=ld.item(), pde=lp.item(),
                                **model.params_float()))
            log("adam", it, w, ld.item(), lp.item())

    # ---------- L-BFGS por bloques, cada bloque con colocación nueva ----------
    n_blocks = 5
    for b in range(n_blocks if n_lbfgs > 0 else 0):
        xi_c, tau_c = sample_colloc(2 * n_colloc, gen)
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=n_lbfgs // n_blocks,
                                history_size=50, tolerance_grad=1e-12, tolerance_change=1e-15,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss, _, _ = losses(xi_c, tau_c, w_pde)
            loss.backward()
            return loss

        opt.step(closure)
        _, ld, lp = losses(xi_c, tau_c, w_pde)
        history.append(dict(phase="lbfgs", it=b, w_pde=w_pde, data=ld.item(), pde=lp.item(),
                            **model.params_float()))
        log("lbfgs", b, w_pde, ld.item(), lp.item())

    # ---------- evaluación final con colocación independiente ----------
    xi_v, tau_v = sample_colloc(4 * n_colloc, gen)
    _, ld, lp_val = losses(xi_v, tau_v, w_pde)
    elapsed = time.time() - t0
    est = model.params_float()
    final = dict(data=ld.item(), chi2=ld.item() * len(r_d), pde_val=lp_val.item())
    if verbose:
        print(f"FINAL ({elapsed:.0f}s): " + " ".join(f"{k}={v:.5g}" for k, v in est.items()) +
              f" | chi2={final['chi2']:.3f} pde_val={final['pde_val']:.2e}", flush=True)
    return model, dict(estimates=est, scenario=scenario, seed=seed, time_s=elapsed,
                       final=final, history=history,
                       config=dict(n_adam=n_adam, n_lbfgs=n_lbfgs, n_colloc=n_colloc,
                                   w_pde_start=w_pde_start, w_pde=w_pde, lr=lr,
                                   lr_param=lr_param, width=width, depth=depth, init=init))


def predict(model, x, t):
    """Evalúa la red en arrays x (μm), t (h). Devuelve r, g como numpy."""
    with torch.no_grad():
        xi = torch.tensor(np.asarray(x, float) / L).reshape(-1, 1)
        tau = torch.tensor(np.asarray(t, float) / T).reshape(-1, 1)
        r, g = model(xi, tau)
    return r.numpy().ravel(), g.numpy().ravel()


def random_init(scenario, seed):
    rng = np.random.default_rng(seed)
    names = ["D", "kr", "kg"] if scenario == 1 else ["Dr", "Dg", "kr", "kg"]
    return {n: float(rng.uniform(200, 1800) if n.startswith("D") else rng.uniform(0.02, 0.15))
            for n in names}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--scenario", type=int, choices=[1, 2], default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_adam", type=int, default=8000)
    ap.add_argument("--n_lbfgs", type=int, default=2500)
    ap.add_argument("--n_colloc", type=int, default=1500)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--w_pde", type=float, default=2e3)
    ap.add_argument("--w_pde_start", type=float, default=10.0)
    ap.add_argument("--random_init", action="store_true",
                    help="inicializa θ al azar dentro de rangos biológicos")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_model", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    df = load_data(a.data) if a.data else load_data()
    init = random_init(a.scenario, a.seed) if a.random_init else None
    model, res = train(df, a.scenario, a.seed, init, a.n_adam, a.n_lbfgs, a.n_colloc,
                       a.w_pde_start, a.w_pde, width=a.width, depth=a.depth)
    res["data"] = a.data or "real"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    if a.save_model:
        Path(a.save_model).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), a.save_model)
