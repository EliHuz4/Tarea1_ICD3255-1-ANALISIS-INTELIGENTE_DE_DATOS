import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def generar_reporte_pdf(tipo_entropia, sufijo_modelo, penalizacion):
    """
    Genera un archivo PDF con los entregables gráficos requeridos y los guarda en 'Graphics/'.
    """
    os.makedirs('Graphics', exist_ok=True)
    ruta_pdf = f"Graphics/Mejor_Modelo_{tipo_entropia}.pdf"
    
    # Cargar datos de los CSV generados previamente por tst.py
    try:
        df_loss = pd.read_csv(f"resultados/convergencia_{sufijo_modelo}.csv")
        with open(f"resultados/metricas_{sufijo_modelo}.csv", "r") as f:
            lineas = f.readlines()
    except FileNotFoundError:
        print(f"Error: No se encontraron los CSV de {sufijo_modelo}. Ejecuta tst.py primero.")
        return
        
    f_train = float(lineas[1].split(',')[1])
    f_test = float(lineas[1].split(',')[2])
    
    # Extraer las matrices de confusión
    cm_train = np.fromstring(lineas[4].strip("[]\n "), sep=' ', dtype=int).reshape(2, 2)
    cm_test = np.fromstring(lineas[6].strip("[]\n "), sep=' ', dtype=int).reshape(2, 2)

    # Iniciar la creación del PDF
    with PdfPages(ruta_pdf) as pdf:
        
        # --- PÁGINA 1: Lista de parámetros ---
        fig = plt.figure(figsize=(8, 6))
        plt.axis('off')
        texto_parametros = (
            f"Lista de Parámetros del Mejor Modelo ({tipo_entropia})\n\n"
            "• Algoritmo de optimización: Descenso de Gradiente con Momentum (mGD)\n"
            "• Tasa de aprendizaje (Learning Rate): 0.05\n"
            "• Momentum: 0.9\n"
            "• Épocas de entrenamiento: 500\n"
            f"• Penalización L2 (Lambda): {penalizacion}\n"
            "• Tamaño de ventana (W): 600 muestras\n"
        )
        plt.text(0.1, 0.6, texto_parametros, fontsize=12, va='top', ha='left')
        pdf.savefig(fig)
        plt.close()

        # --- PÁGINA 2: Curva de convergencia ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df_loss['Epoca'], df_loss['Loss_Entropia_Cruzada'], color='#1f77b4', linewidth=2)
        ax.set_title(f'Curva de Convergencia (mGD) - {tipo_entropia}', fontsize=14)
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Loss (Entropía Cruzada)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        pdf.savefig(fig)
        plt.close()

        # --- PÁGINA 3: Matrices de Confusión ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
        axes[0].set_title('Matriz de Confusión - Train')
        axes[0].set_xlabel('Predicción')
        axes[0].set_ylabel('Real')
        
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
        axes[1].set_title('Matriz de Confusión - Test')
        axes[1].set_xlabel('Predicción')
        axes[1].set_ylabel('Real')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # --- PÁGINA 4: F-scores ---
        fig, ax = plt.subplots(figsize=(6, 5))
        etiquetas = ['Train', 'Test']
        valores = [f_train, f_test]
        barras = ax.bar(etiquetas, valores, color=['#4C72B0', '#55A868'])
        ax.set_ylim(0, 1.1)
        ax.set_title(f'F-scores del Modelo - {tipo_entropia}', fontsize=14)
        ax.set_ylabel('Puntuación F1', fontsize=12)
        
        for barra in barras:
            altura = barra.get_height()
            ax.text(barra.get_x() + barra.get_width()/2., altura + 0.02,
                    f'{altura:.4f}', ha='center', va='bottom', fontsize=11)
            
        pdf.savefig(fig)
        plt.close()

def generar_explicacion_md():
    """Genera un archivo Markdown en la carpeta Graphics explicando los resultados."""
    contenido = """# Análisis de Gráficos de Diagnóstico de Rodamientos (MSE-LR)

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
"""
    with open("Graphics/explicacion_resultados.md", "w", encoding="utf-8") as f:
        f.write(contenido)

if __name__ == "__main__":
    print("Generando PDFs y archivo explicativo en la carpeta 'Graphics'...")
    
    # Generar gráficos para MPE y MDE
    generar_reporte_pdf('MPE', 'MPE_Estandar', penalizacion=0.0)
    generar_reporte_pdf('MDE', 'MDE_Penalizada', penalizacion=0.1)
    
    # Generar archivo Markdown
    generar_explicacion_md()
    
    print("¡Proceso finalizado! Revisa la carpeta 'Graphics'.")