# Análisis de Gráficos de Diagnóstico de Rodamientos (MSE-LR)

Este documento explica la interpretación de los gráficos generados para evaluar el modelo de Regresión Logística basado en Entropía Multiescala.

## 1. Curva de Convergencia
Muestra cómo disminuye el error (pérdida o *Loss*) a medida que el algoritmo de Descenso de Gradiente con Momentum ($mGD$) avanza a lo largo de las 500 épocas. 
* Una curva que desciende de forma suave y se estabiliza cerca de cero indica que el algoritmo está aprendiendo correctamente a separar las características de señales normales y con fallos.
* La optimización de la función de Entropía Cruzada es clave para asegurar la precisión del diagnóstico.

## 2. Matrices de Confusión (Train / Test)
Representan el desempeño de clasificación del modelo separando falsos positivos, falsos negativos, verdaderos positivos y verdaderos negativos.
* **Eje 'Real' vs 'Predicción':** Un modelo perfecto tendrá valores altos únicamente en la diagonal principal (esquina superior izquierda a inferior derecha).
* Las pequeñas diferencias entre la matriz de *Train* y *Test* nos ayudan a evaluar si el modelo generaliza bien con datos que no ha visto (evitando el sobreajuste o *overfitting*).

## 3. F-Scores
El F-score (o F1-score) es la media armónica entre la precisión y la exhaustividad (*recall*). 
* Un valor cercano a 1.0 (o 100%) indica que el modelo es altamente fiable y no está sesgado por un desbalance de clases.
* Es la métrica final que confirma si el estado de salud del rodamiento se está diagnosticando de forma confiable.
