# Datos FUCCI

Fuente: Simpson, Baker, Vittadello y Maclaren (2020), *Practical parameter identifiability for spatio-temporal models of cell invasion*, DOI: https://doi.org/10.1098/rsif.2020.0055.
Material suplementario: https://doi.org/10.6084/m9.figshare.c.4862265.v1.

`datos_reorganizados.csv` es una transcripción numérica de la primera hoja del archivo de trabajo `datos_reorganizados.xlsx`, ordenada por tiempo y posición. No se generaron observaciones nuevas ni se imputaron valores. Se conserva precisión de 17 cifras significativas al exportar. El Excel original no se incluye para evitar duplicaciones y metadatos de edición.

| Columna | Significado |
|---|---|
| x_um | Posición, micrómetros; 24 posiciones de 0 a 1242, paso 54 |
| t_h | Tiempo, horas: 0, 16, 32, 48 |
| conteo_rojas | Conteo observado del grupo G1 |
| conteo_verdes | Conteo observado del grupo S/G2/M |
| r_obs | Densidad roja: conteo / 377.018 |
| g_obs | Densidad verde: conteo / 377.018 |

Hay 96 filas posición-tiempo y dos poblaciones. El ajuste excluye t=0: 72 pares o 144 residuos escalares. La condición inicial es un ajuste de cosenos hasta grado 8, fijado durante la inferencia. Los datos y materiales de terceros conservan sus condiciones originales; la publicación del repositorio no les asigna una licencia nueva.
