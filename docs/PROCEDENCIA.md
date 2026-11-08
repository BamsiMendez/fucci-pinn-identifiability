# Procedencia y decisiones editoriales

Preparación: 8 de noviembre de 2026. Autores del proyecto: Bamsi Mendez Posada y Sergio Alejandro Villada Arias.

## Referencia científica

Se conserva el PDF identificado como VERSION_FINAL, presentado el 6 de noviembre. Sus cifras se respaldan con las tablas del paquete de verificación e identificabilidad. Aunque estas tablas estaban reunidas en una carpeta de la primera propuesta gráfica, sus valores coinciden con los del póster final.

## Selección

| Elemento original | Tratamiento |
|---|---|
| Póster VERSION_FINAL | Una copia, nombre estable, bytes idénticos |
| common.py, fit_fd.py, pinn.py, run_study.py del paquete de trabajo | src/; rutas portables, CSV, comprobaciones de fallo y creación de salidas |
| Notebooks de verificación e intervalos | Dos notebooks; solver compartido, sin base64, sin salidas, sin descargas automáticas |
| Excel reorganizado | CSV numérico; se preservan observaciones y orden; se eliminan metadatos de libro |
| Tablas de verificación, perfiles y réplicas finales | results/published/; bytes originales conservados |
| Figuras explicativas | Selección PNG para lectura y PDF para el informe |
| Guía de estudio | Adaptación científica: fundamentos, métodos, resultados y limitaciones; se retiran guiones, ejercicios y preparación oral |
| Análisis clásico inicial y notebooks exploratorios | Excluidos: dominio y condición inicial diferentes de la versión final |
| Conversaciones, contexto, bibliografía PDF, versiones gráficas y auxiliares | Excluidos |

## Transformaciones del código

La carga usa el CSV por defecto; los Excel siguen siendo admitidos si se instala openpyxl por separado. Se elimina una ruta personal de Linux. Se añaden comprobaciones de éxito del integrador y del ajuste, validación de escenario por CLI y creación de directorios de salida. No se cambian ecuaciones, arquitectura, cotas, semilla ni presupuestos predeterminados de entrenamiento.

Los notebooks guardan nuevos resultados fuera de las tablas archivadas. El `configuracion.json` original conserva los hashes del Excel y CSV de entrada de las ejecuciones históricas; no se reescribe para fingir que esas corridas usaron el CSV nuevo. `manifest_sha256.json` registra los archivos de esta edición.

## Alcance

No se certifican todas las configuraciones efectivas ni se distribuyen todos los pesos neuronales de las corridas históricas. Reproducir la estructura del experimento no implica igualdad bit a bit con el entrenamiento original. La variabilidad entre semillas no se presenta como intervalo de confianza. Las funcionalidades sintéticas del controlador no se presentan como resultados ya obtenidos.

La documentación y reorganización se prepararon con asistencia de IA y se contrastaron con los archivos del proyecto. Los aportes científicos y las referencias mantienen su atribución.
