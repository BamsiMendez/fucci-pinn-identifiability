"""Resume las tablas archivadas; no recalcula ni entrena modelos."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "results/published"

def summarize():
    ci = pd.read_csv(P / "comparacion_intervalos.csv")
    scores = pd.read_csv(P / "comparacion_escenarios.csv")
    s1 = pd.read_csv(P / "real_s1.csv")
    v2 = pd.read_csv(P / "verificacion_s2.csv")
    result = {
        "s1_mle": dict(zip(ci.query("escenario == 1").parametro, ci.query("escenario == 1").estimacion)),
        "s1_D_interval": ci.query("escenario == 1 and parametro == 'D'")[["inferior", "superior"]].iloc[0].tolist(),
        "s1_pinn_mean": {n: float(s1["pinn_" + n].mean()) for n in ["D", "kr", "kg"]},
        "s1_pinn_sd": {n: float(s1["pinn_" + n].std(ddof=1)) for n in ["D", "kr", "kg"]},
        "s1_restarted_runs": int((s1.n_attempts > 1).sum()),
        "s2_outside_mle_bounds": int(v2.fuera_cotas.sum()),
        "s1_rmse": float(.05 * np.sqrt(scores.loc[scores.escenario == 1, "chi2"].iloc[0] / 144)),
        "delta_AIC_s2": float(scores.loc[scores.escenario == 2, "delta_AIC"].iloc[0]),
    }
    return result

if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2, ensure_ascii=False))
