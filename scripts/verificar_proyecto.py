"""Verifica integridad y coherencia científica básica, sin entrenar redes."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from common import load_data, data_arrays, FDSolver

def main():
    manifest=json.loads((ROOT/'manifest_sha256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Archivo modificado respecto de la edición archivada: {name}')
    df=load_data()
    xs,ts,R,G=data_arrays(df)
    assert len(df)==96 and not df.duplicated(['x','t']).any()
    assert np.array_equal(xs,np.arange(24)*54)
    assert np.array_equal(ts,[0,16,32,48])
    assert np.isfinite(df.to_numpy()).all()
    assert np.all((R>=0)&(G>=0)&(R+G<=1))
    raw=pd.read_csv(ROOT/'data/datos_reorganizados.csv')
    # El libro de trabajo conserva densidades redondeadas (precisión ~1e-9).
    assert np.allclose(raw.r_obs,raw.conteo_rojas/377.018,atol=1e-9,rtol=0)
    assert np.allclose(raw.g_obs,raw.conteo_verdes/377.018,atol=1e-9,rtol=0)
    fd=FDSolver(xs,R[0],G[0])
    P=ROOT/'results/published'
    for sc in [1,2]:
        mle=pd.read_csv(P/f'mle_s{sc}.csv').set_index('parametro').estimacion
        dr,dg=(mle.D,mle.D) if sc==1 else (mle.Dr,mle.Dg)
        r,g=fd.at_obs(dr,dg,mle.kr,mle.kg)
        assert np.isfinite(r).all() and np.isfinite(g).all()
        stored=pd.read_csv(P/f'residuos_s{sc}.csv')
        for pop,expected in [('G1',r),('S/G2/M',g)]:
            actual=stored[stored.poblacion==pop].sort_values(['t','x']).predicho.to_numpy()
            np.testing.assert_allclose(actual,expected[1:].ravel(),atol=2e-7,rtol=1e-5)
    for file in (ROOT/'src').glob('*.py'):
        ast.parse(file.read_text(encoding='utf-8'))
    for file in (ROOT/'notebooks').glob('*.ipynb'):
        nb=json.loads(file.read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                assert not cell['outputs'] and cell['execution_count'] is None
                ast.parse(''.join(cell['source']))
    print(f'OK: {len(manifest)} hashes; 96 observaciones; dos soluciones FD coherentes con tablas; código y notebooks válidos.')

if __name__=='__main__':
    main()
