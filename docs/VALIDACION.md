# Validación de esta edición

Fecha: 8 de noviembre de 2026.

## Comprobaciones realizadas

- Póster conservado idéntico byte por byte respecto del PDF final seleccionado.
- Datos: 96 filas posición-tiempo, cuatro tiempos, 24 posiciones, dos poblaciones, claves únicas y densidades coherentes con la normalización (tolerancia 1e-9 por redondeo del libro).
- Solución FD recalculada con los parámetros archivados en ambos escenarios. Predicciones coherentes con las tablas publicadas, usando tolerancias `atol=2e-7`, `rtol=1e-5`.
- Resumen de cifras: medias y desviaciones PINN, dos reinicios, cuatro vectores fuera de cotas, RMSE e intervalo de D coinciden con la evidencia presentada.
- Notebook 01 ejecutado completo desde archivos locales: MLE desde varios inicios, verificación de veinte vectores PINN, residuos y sensibilidad a la condición inicial.
- Notebook 02: código de todas las celdas analizado sintácticamente; configuración, carga y MLE de ambos escenarios ejecutados. Se calculó la rejilla completa del perfil de D compartida; el recálculo total de perfiles e intervalos se interrumpió por su coste. No se afirma una ejecución completa de este notebook en esta edición.
- PINN: prueba breve de dos pasos Adam y evaluación diferencial, sin L-BFGS. Comprueba que carga y entrenamiento funcionan; no equivale a reproducir los experimentos ni a validar sus resultados científicos.
- Enlaces relativos de Markdown revisados, notebooks sin salidas ni datos incrustados, y código sin rutas personales absolutas.
- Informe compilado a 29 páginas y todas las páginas renderizadas e inspeccionadas. Sin advertencias de desbordamiento horizontal ni referencias indefinidas en la compilación final.

## Límites

No se repitieron los veinte entrenamientos completos ni se evaluó cobertura. No se reconstruyeron todos los campos neuronales originales por falta de todos los pesos y registros. Las tablas `results/published/` siguen siendo las de las ejecuciones archivadas; las nuevas salidas de comprobación no las sustituyen ni se incluyen en la selección pública.

`scripts/verificar_proyecto.py` permite repetir las verificaciones de integridad, estructura y predicciones FD. `manifest_sha256.json` registra los archivos de esta edición; modificar intencionalmente un archivo obliga a actualizar el manifiesto antes de usarlo como referencia de integridad.
