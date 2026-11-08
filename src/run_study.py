"""Estudio de recuperación de parámetros con la PINN.

Modos:
  real      : entrena la PINN K veces (semillas distintas) sobre los datos experimentales.
  synthetic : genera K conjuntos sintéticos con θ verdadero conocido (solución FD + ruido
              N(0, σ²) en t>0, CI = datos reales en t=0), y en cada uno ajusta PINN y MLE-FD.

Las ejecuciones corren en paralelo (1 hilo por proceso) y se reanudan si se interrumpen
(se omiten las réplicas cuyo JSON ya existe). Resultado final: un CSV en results/.

Ejemplos:
  python run_study.py --mode real --scenario 1 --reps 20 --workers 10
  python run_study.py --mode synthetic --scenario 1 --reps 50 --workers 10 --noise 0.05
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from common import load_data, data_arrays, FDSolver, TIMES

PDE_TOL = 1e-5       # umbral del residuo EDP (validación) para declarar convergencia
MAX_ATTEMPTS = 3

TRUE_DEFAULT = {1: {"D": 696.2, "kr": 0.02914, "kg": 0.07238},
                2: {"Dr": 600.9, "Dg": 880.2, "kr": 0.02937, "kg": 0.07212}}


def make_synthetic(theta, scenario, noise, seed):
    df = load_data()
    xs, ts, R, G = data_arrays(df)
    fd = FDSolver(xs, R[0], G[0])
    Dr, Dg = (theta["D"], theta["D"]) if scenario == 1 else (theta["Dr"], theta["Dg"])
    ro, go = fd.at_obs(Dr, Dg, theta["kr"], theta["kg"], TIMES)
    rng = np.random.default_rng(10_000 + seed)
    ro[1:] += rng.normal(0, noise, ro[1:].shape)
    go[1:] += rng.normal(0, noise, go[1:].shape)
    ro[0], go[0] = R[0], G[0]
    rows = [(x, t, ro[i, j], go[i, j]) for i, t in enumerate(ts) for j, x in enumerate(xs)]
    return pd.DataFrame(rows, columns=["x", "t", "r", "g"])


def job(kind, scenario, rep, theta, noise, outdir, pinn_kw):
    import torch
    torch.set_num_threads(1)
    from pinn import train, random_init
    from fit_fd import fit_mle
    out = os.path.join(outdir, f"{kind}_s{scenario}_rep{rep:03d}.json")
    old = json.load(open(out)) if os.path.exists(out) else None
    if old is not None and "attempts" in old:
        return out
    if kind == "real":
        df = load_data()
    else:
        df = make_synthetic(theta, scenario, noise, rep)

    # Reintentos: un entrenamiento se declara fallido si el residuo de la EDP en puntos de
    # validación supera PDE_TOL (criterio que NO usa el valor verdadero ni el MLE).
    # Se prueba una nueva inicialización aleatoria de θ, hasta MAX_ATTEMPTS.
    attempts = []
    if old is not None:   # resultado previo (sin reintentos): se reutiliza como intento 0
        attempts.append(dict(estimates=old["pinn"], final=old["pinn_final"],
                             time_s=old["pinn_time_s"], history=old.get("history"),
                             config=old["config"]))
    best_model = None
    while True:
        a = len(attempts)
        if a > 0 and attempts[-1]["final"]["pde_val"] <= PDE_TOL:
            break
        if a >= MAX_ATTEMPTS:
            break
        # inicialización aleatoria de θ (distinta por réplica e intento)
        model, res = train(df, scenario=scenario, seed=rep + 100 * a,
                           init=random_init(scenario, 1000 + rep + 100 * a), verbose=False, **pinn_kw)
        attempts.append(res)
        best_model = model
    ok = [r for r in attempts if r["final"]["pde_val"] <= PDE_TOL]
    # resultado elegido: el último intento válido; si ninguno lo es, el de menor residuo
    chosen = ok[-1] if ok else min(attempts, key=lambda r: r["final"]["pde_val"])
    if kind == "real" and best_model is not None and chosen is attempts[-1]:
        torch.save(best_model.state_dict(), out.replace(".json", ".pt"))
    rec = dict(kind=kind, scenario=scenario, rep=rep, noise=noise if kind == "synthetic" else None,
               true=theta if kind == "synthetic" else None, pinn=chosen["estimates"],
               pinn_final=chosen["final"], pinn_time_s=sum(r["time_s"] for r in attempts),
               n_attempts=len(attempts), converged=bool(ok),
               attempts=[dict(estimates=r["estimates"], final=r["final"], init=r["config"]["init"])
                         for r in attempts],
               history=chosen.get("history"), config=chosen["config"])
    if kind == "synthetic":
        xs, ts, R, G = data_arrays(df)
        mle, se, chi2 = fit_mle(xs, R, G, scenario)
        rec.update(mle=mle, mle_se=se, mle_chi2=chi2)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    return out


def collect(outdir, kind, scenario):
    rows = []
    for fn in sorted(os.listdir(outdir)):
        if not (fn.startswith(f"{kind}_s{scenario}_rep") and fn.endswith(".json")):
            continue
        r = json.load(open(os.path.join(outdir, fn)))
        row = dict(rep=r["rep"], noise=r["noise"], pinn_time_s=r["pinn_time_s"],
                   n_attempts=r.get("n_attempts", 1), converged=r.get("converged"),
                   pinn_chi2=r["pinn_final"]["chi2"], pinn_pde_val=r["pinn_final"]["pde_val"])
        a0 = r["attempts"][0] if r.get("attempts") else None
        if a0:   # resultado del primer intento (sin reintento), útil para medir la tasa de fallo
            for k, v in a0["estimates"].items():
                row[f"first_{k}"] = v
            row["first_pde_val"] = a0["final"]["pde_val"]
        for k, v in r["config"]["init"].items():
            row[f"init_{k}"] = v
        for k, v in r["pinn"].items():
            row[f"pinn_{k}"] = v
        if r.get("true"):
            for k, v in r["true"].items():
                row[f"true_{k}"] = v
            for k, v in r["mle"].items():
                row[f"mle_{k}"] = v
            row["mle_chi2"] = r["mle_chi2"]
            if r.get("mle_se"):
                for k, v in r["mle_se"].items():
                    row[f"mle_se_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "synthetic"], default="synthetic")
    ap.add_argument("--scenario", type=int, choices=[1, 2], default=1)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--true", default=None, help='JSON, p.ej. \'{"D":700,"kr":0.03,"kg":0.07}\'')
    ap.add_argument("--w_pde", type=float, default=None)
    ap.add_argument("--outdir", default="results/runs")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    theta = json.loads(a.true) if a.true else TRUE_DEFAULT[a.scenario]
    pinn_kw = {} if a.w_pde is None else {"w_pde": a.w_pde}
    reps = range(a.start, a.start + a.reps)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(job, a.mode, a.scenario, r, theta, a.noise, a.outdir, pinn_kw): r for r in reps}
        for f in as_completed(futs):
            try:
                print("ok", f.result(), flush=True)
            except Exception as e:
                print("FALLÓ rep", futs[f], repr(e), flush=True)
    df = collect(a.outdir, a.mode, a.scenario)
    os.makedirs("results", exist_ok=True)
    csv = f"results/{a.mode}_s{a.scenario}.csv"
    df.to_csv(csv, index=False)
    print(f"\nGuardado {csv} ({len(df)} filas)")
    print(df.describe().T.to_string())
