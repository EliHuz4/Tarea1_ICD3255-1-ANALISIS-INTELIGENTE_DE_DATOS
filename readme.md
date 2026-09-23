# Tarea #1: Diagnóstico de Fallos en Rodamientos (MSE-LR)

**Estudiantes:** Anais Diaz - Cristian Gallardo - Matías Salas

**Profesor:** Prof. Nibaldo Rodríguez

---

## Descripción del Proyecto
Este repositorio contiene la solución a la Tarea #1 enfocada en el diagnóstico de fallos incipientes en rodamientos de motores eléctricos de inducción para mantenimiento predictivo[cite: 1]. El método propuesto se valida utilizando el Dataset del Bearing Data Center de la Case Western Reserve University (CWRU) empleando la señal del acelerómetro del extremo de acoplamiento (`DE_time`)[cite: 1].

El proyecto implementa la extracción de características mediante Entropía Multi-escala de Permutación (MPE) y Entropía Multi-escala de Dispersión (MDE) particionando la señal en ventanas de tamaño *W*[cite: 1]. Para la clasificación, se ajustan los coeficientes de un modelo de Regresión Logística utilizando la función de Entropía Cruzada (estándar y penalizada) optimizada a través del algoritmo de Descenso del Gradiente con Momentum ($mGD$)[cite: 1].

## Estructura del Repositorio

El proyecto está organizado según los módulos solicitados en los entregables[cite: 1]:

* **`data/`**: Contiene los archivos `.mat` del dataset CWRU, organizados internamente en las subcarpetas `normal/` y `fallos/`.
* **`resultados/`**: Directorio donde se exportan automáticamente los entregables numéricos (`.csv`) y gráficos (`.pdf`) tras la ejecución del código.
* **`utility.py`**: Módulo que contiene las funciones base para la carga de datos `.mat`, la partición de la señal en ventanas y el cálculo de la entropía MPE y MDE usando la librería *EntropyHub*[cite: 1].
* **`train.py`**: Script que contiene la preparación de datos y la implementación orientada a objetos del modelo `LogisticRegressionmGD` (Regresión Logística con $mGD$ y Entropía Cruzada Penalizada)[cite: 1].
* **`tst.py`**: Script principal de evaluación. Divide los datos en *train/test*, entrena los distintos modelos requeridos y exporta los coeficientes, curvas de convergencia, matrices de confusión y F-scores en formato `.csv`[cite: 1].
* **`plot.py`**: Módulo de visualización que lee los datos `.csv` generados y construye los reportes gráficos finales en formato `.pdf` para el mejor modelo de MPE y MDE[cite: 1].

## Requisitos y Dependencias

El código está escrito en Python 3. Asegúrese de instalar las siguientes dependencias antes de la ejecución:

```
pip install numpy scipy pandas scikit-learn matplotlib seaborn EntropyHub tqdm
```

## Instrucciones de Ejecución

Para reproducir los resultados y generar los entregables, los scripts deben ejecutarse en el siguiente orden secuencial desde la raíz del proyecto:

1. **Ejecutar las pruebas y extracciones numéricas:**
   ```
   python tst.py
   ```
   *Nota: Este script escaneará automáticamente el directorio `data/`, extraerá las entropías, entrenará los modelos y poblará la carpeta `resultados/` con los archivos `.csv` correspondientes.*

2. **Generar los reportes gráficos (PDF):**
   ```
   python plot.py
   ```
   *Nota: Este script leerá los `.csv` generados en el paso anterior y creará los archivos PDF con la matriz de confusión, los F-scores, la curva de convergencia y la lista de parámetros de los mejores modelos.*

## Entregables Incluidos

Al finalizar la ejecución, el proyecto generará los siguientes archivos requeridos por la evaluación[cite: 1]:

**Archivos Python (Código Fuente):**
* `train.py`, `utility.py`, `tst.py`, `plot.py`[cite: 1]

**Archivos de Resultados Numéricos (.csv):**
* `coeficientes_[modelo].csv`: Coeficientes de regresión ajustados[cite: 1].
* `convergencia_[modelo].csv`: Valores de convergencia del algoritmo mGD (Loss por época)[cite: 1].
* `metricas_[modelo].csv`: Matriz de confusión y F-scores para conjuntos de *train* y *test*[cite: 1].

**Archivos Gráficos (.pdf):**
* `Mejor_Modelo_MPE.pdf`: Reporte gráfico del modelo usando Permutation Entropy[cite: 1].
* `Mejor_Modelo_MDE.pdf`: Reporte gráfico del modelo usando Dispersion Entropy[cite: 1].
*(Ambos PDF incluyen la lista de parámetros, la curva de convergencia, las matrices de confusión y los F-scores de train/test).*