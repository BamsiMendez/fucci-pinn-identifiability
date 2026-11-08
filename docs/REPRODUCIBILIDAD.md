# Reproducción

Ejecutar los comandos desde la raíz del repositorio. Se recomienda Python 3.11 o posterior. Las dependencias tienen rangos compatibles, no constituyen un entorno congelado de las corridas originales.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/resumir_resultados.py
python scripts/verificar_proyecto.py
```

Estos dos últimos comandos comprueban archivos y resumen evidencia archivada sin reentrenar modelos. Para abrir los notebooks:

```bash
jupyter lab
```

`notebooks/01_Verificacion_PINN_FUCCI.ipynb` recalcula MLE, verificación FD de vectores PINN, residuos y sensibilidad al estado inicial. `notebooks/02_Perfiles_Intervalos_FUCCI.ipynb` recalcula perfiles, Fisher y AIC. Son independientes entre sí y escriben en `results/generated/`. Requieren múltiples integraciones de la EDP; el segundo puede tardar considerablemente.

## Ajuste clásico individual

```bash
python src/fit_fd.py --scenario 1
python src/fit_fd.py --scenario 2
```

## Entrenamiento PINN opcional

```bash
python -m pip install -r requirements-pinn.txt
python src/run_study.py --mode real --scenario 1 --reps 10 --workers 1 --outdir results/generated/runs_s1
python src/run_study.py --mode real --scenario 2 --reps 10 --workers 1 --outdir results/generated/runs_s2
```

Un trabajador limita la demanda de memoria. Cada entrenamiento puede durar varios minutos o más; diez réplicas y reinicios aumentan el coste. El controlador guarda los JSON y pesos en `--outdir` y los CSV agregados en `results/real_s1.csv` y `results/real_s2.csv`; no sustituye `results/published/`. No reutilizar una carpeta de réplicas para configuraciones diferentes: el controlador reanuda resultados existentes.

La parametrización neuronal impone positividad, pero no las cotas de MLE. Comparar errores requiere evaluar los parámetros PINN con el solver FD bajo las mismas condiciones. Sin todos los pesos originales no puede reconstruirse exactamente el campo de cada entrenamiento archivado.

## Informe

```bash
cd docs
pdflatex Informe_PINNs_FUCCI.tex
pdflatex Informe_PINNs_FUCCI.tex
```

Los gráficos ya están incluidos. El PDF es una edición documental basada en los resultados existentes, sin nuevas afirmaciones de entrenamiento o cobertura.
