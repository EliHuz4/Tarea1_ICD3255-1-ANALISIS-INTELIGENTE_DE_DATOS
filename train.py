import os
import numpy as np
from tqdm import tqdm
from utility import cargar_senal_cwru, particionar_ventanas, calcular_mpe, calcular_mde

def preparar_datos(ruta_base, tamano_ventana=600, tipo_entropia='MPE'):
    """
    Escanea las carpetas para extraer las entropías MPE o MDE.
    Asigna etiqueta 0 a normal y 1 a fallos.
    """
    X = []
    y = []
    carpetas = {'normal': 0, 'fallos': 1}
    
    for carpeta, etiqueta in carpetas.items():
        ruta_carpeta = os.path.join(ruta_base, carpeta)
        if not os.path.exists(ruta_carpeta):
            print(f"Advertencia: No se encontró la carpeta {ruta_carpeta}")
            continue
            
        archivos_mat = [f for f in os.listdir(ruta_carpeta) if f.endswith('.mat')]
        
        # Envolvemos la lista de archivos con tqdm para la barra de progreso
        for archivo in tqdm(archivos_mat, desc=f"Procesando {carpeta} ({tipo_entropia})"):
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            senal = cargar_senal_cwru(ruta_archivo)
            ventanas = particionar_ventanas(senal, tamano_ventana)
            
            if tipo_entropia == 'MPE':
                features = calcular_mpe(ventanas)
            elif tipo_entropia == 'MDE':
                features = calcular_mde(ventanas)
                
            X.append(features)
            y.extend([etiqueta] * len(features))
                
    return np.vstack(X), np.array(y)

class RegresionLogistica_mGD:
    def __init__(self, lr=0.01, momentum=0.9, epocas=500, lambda_penalizacion=0.0):
        self.lr = lr
        self.momentum = momentum
        self.epocas = epocas
        self.lambda_penalizacion = lambda_penalizacion
        self.w = None
        self.b = None
        self.historial_loss = []

    def sigmoide(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def calcular_costo(self, y, y_pred):
        m = len(y)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        costo = -(1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        if self.lambda_penalizacion > 0:
            costo += (self.lambda_penalizacion / (2 * m)) * np.sum(np.square(self.w))
        return costo

    def fit(self, X, y):
        m, n_caracteristicas = X.shape
        self.w = np.zeros(n_caracteristicas)
        self.b = 0
        
        vw = np.zeros(n_caracteristicas)
        vb = 0

        for _ in range(self.epocas):
            z = np.dot(X, self.w) + self.b
            y_pred = self.sigmoide(z)
            
            self.historial_loss.append(self.calcular_costo(y, y_pred))

            # Cálculo de Gradientes
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            if self.lambda_penalizacion > 0:
                dw += (self.lambda_penalizacion / m) * self.w
            db = (1/m) * np.sum(y_pred - y)

            vw = self.momentum * vw + self.lr * dw
            vb = self.momentum * vb + self.lr * db

            self.w -= vw
            self.b -= vb

    def predict_proba(self, X):
        return self.sigmoide(np.dot(X, self.w) + self.b)

    def predict(self, X, umbral=0.5):
        return (self.predict_proba(X) >= umbral).astype(int)

# --- Sección de prueba local ---
if __name__ == "__main__":
    print("Extrayendo características MPE de los directorios 'normal' y 'fallos'...")
    X_prueba, y_prueba = preparar_datos('data', tamano_ventana=600, tipo_entropia='MPE')
    
    print(f"Dimensiones de X (Características): {X_prueba.shape}")
    print(f"Dimensiones de y (Etiquetas): {y_prueba.shape}")