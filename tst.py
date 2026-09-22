import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from train import preparar_datos, RegresionLogistica_mGD

def evaluar_y_guardar(tipo_entropia, penalizacion=0.0):
    """
    Entrena el modelo y exporta los resultados a CSV (coeficientes, convergencia, métricas).
    """
    sufijo = f"{tipo_entropia}_Penalizada" if penalizacion > 0 else f"{tipo_entropia}_Estandar"
    print(f"\n--- Evaluando modelo: {sufijo} ---")
    
    # 1. Cargar datos
    X, y = preparar_datos('data', tamano_ventana=600, tipo_entropia=tipo_entropia)
    
    # 2. Partición Train/Test (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 3. Entrenamiento con mGD
    modelo = RegresionLogistica_mGD(lr=0.05, momentum=0.9, epocas=500, lambda_penalizacion=penalizacion)
    modelo.fit(X_train, y_train)
    
    # 4. Predicciones
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # 5. Métricas (F-score y Matriz de Confusión)
    f_score_train = f1_score(y_train, y_pred_train)
    f_score_test = f1_score(y_test, y_pred_test)
    
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)
    
    print(f"F-score Train: {f_score_train:.4f} | F-score Test: {f_score_test:.4f}")
    
    # 6. Exportación a CSV
    os.makedirs('resultados', exist_ok=True)
    
    # Exportar Coeficientes
    df_coef = pd.DataFrame({
        'Parametro': [f'w_escala_{i+1}' for i in range(len(modelo.w))] + ['bias'],
        'Valor': list(modelo.w) + [modelo.b]
    })
    df_coef.to_csv(f'resultados/coeficientes_{sufijo}.csv', index=False)
    
    # Exportar Convergencia
    df_loss = pd.DataFrame({
        'Epoca': range(1, len(modelo.historial_loss) + 1),
        'Loss_Entropia_Cruzada': modelo.historial_loss
    })
    df_loss.to_csv(f'resultados/convergencia_{sufijo}.csv', index=False)
    
    # Exportar Métricas (F-score y Matrices)
    with open(f'resultados/metricas_{sufijo}.csv', 'w') as f:
        f.write("Metrica,Train,Test\n")
        f.write(f"F-score,{f_score_train},{f_score_test}\n")
        f.write("\nMatriz de Confusion Train (TN FP FN TP)\n")
        f.write(f"{cm_train.ravel()}\n")
        f.write("Matriz de Confusion Test (TN FP FN TP)\n")
        f.write(f"{cm_test.ravel()}\n")

    return modelo

if __name__ == "__main__":
    # Evaluar MPE (Entregables 1)
    evaluar_y_guardar('MPE', penalizacion=0.0) # Entropía cruzada normal
    evaluar_y_guardar('MPE', penalizacion=0.1) # Entropía cruzada penalizada
    
    # Evaluar MDE (Entregables 2)
    evaluar_y_guardar('MDE', penalizacion=0.0)
    evaluar_y_guardar('MDE', penalizacion=0.1)
    
    print("\n¡Todos los archivos CSV fueron generados con éxito en la carpeta 'resultados'!")