import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def generar_reporte_pdf(tipo_entropia, sufijo_modelo, penalizacion):

    ruta_pdf = f"resultados/Mejor_Modelo_{tipo_entropia}.pdf"
    
    df_loss = pd.read_csv(f"resultados/convergencia_{sufijo_modelo}.csv")
    
    with open(f"resultados/metricas_{sufijo_modelo}.csv", "r") as f:
        lineas = f.readlines()
        
    f_train = float(lineas[1].split(',')[1])
    f_test = float(lineas[1].split(',')[2])
    
    
    cm_train = np.fromstring(lineas[4].strip("[]\n "), sep=' ', dtype=int).reshape(2, 2)
    cm_test = np.fromstring(lineas[6].strip("[]\n "), sep=' ', dtype=int).reshape(2, 2)

    with PdfPages(ruta_pdf) as pdf:
        
        # --- PÁGINA 1 ---
        fig = plt.figure(figsize=(8, 6))
        plt.axis('off')
        texto_parametros = (
            f"Lista de Parámetros del Mejor Modelo ({tipo_entropia})\n\n"
            "• Algoritmo de optimización: Descenso de Gradiente con Momentum (mGD)\n"
            f"• Tasa de aprendizaje (Learning Rate): 0.05\n"
            f"• Momentum: 0.9\n"
            f"• Épocas de entrenamiento: 500\n"
            f"• Penalización L2 (Lambda): {penalizacion}\n"
            f"• Tamaño de ventana (W): 600 muestras\n"
        )
        plt.text(0.1, 0.6, texto_parametros, fontsize=12, va='top', ha='left')
        pdf.savefig(fig)
        plt.close()

        # --- PÁGINA 2 ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df_loss['Epoca'], df_loss['Loss_Entropia_Cruzada'], color='#1f77b4', linewidth=2)
        ax.set_title(f'Curva de Convergencia (mGD) - {tipo_entropia}', fontsize=14)
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Loss (Entropía)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        pdf.savefig(fig)
        plt.close()

        # --- PÁGINA 3 ---
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

        # --- PÁGINA 4 ---
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

if __name__ == "__main__":
    print("Generando PDFs de los entregables...")
    
    generar_reporte_pdf('MPE', 'MPE_Estandar', penalizacion=0.0)
    generar_reporte_pdf('MDE', 'MDE_Penalizada', penalizacion=0.1)
    
    print("¡PDFs generados exitosamente en la carpeta 'resultados'!")