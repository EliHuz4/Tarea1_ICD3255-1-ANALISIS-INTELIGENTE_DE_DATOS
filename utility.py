import scipy.io
import numpy as np
import EntropyHub as EH
import os
import contextlib

def cargar_senal_cwru(ruta_archivo):
    mat_data = scipy.io.loadmat(ruta_archivo)
    claves_de = [key for key in mat_data.keys() if 'DE_time' in key]
    if not claves_de:
        raise ValueError("No se encontró la variable DE_time en el archivo.")
    senal = mat_data[claves_de[0]].flatten()
    return senal

def particionar_ventanas(senal, tamano_ventana):
    num_ventanas = len(senal) // tamano_ventana
    senal_recortada = senal[:num_ventanas * tamano_ventana]
    ventanas = np.reshape(senal_recortada, (num_ventanas, tamano_ventana))
    return ventanas

def calcular_mpe(ventanas, m=3, tau=1, escalas=5):
    caracteristicas = []
    Mobj = EH.MSobject('PermEn', m=m, tau=tau)
    
    for ventana in ventanas:
        # Silencia los puntos de EntropyHub redirigiendo la salida estándar
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            entropias, _ = EH.MSEn(ventana, Mobj, Scales=escalas)
        caracteristicas.append(entropias)
        
    return np.array(caracteristicas)

def calcular_mde(ventanas, m=2, tau=1, c=3, escalas=5):
    caracteristicas = []
    Mobj = EH.MSobject('DispEn', m=m, tau=tau, c=c)
    
    for ventana in ventanas:
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            entropias, _ = EH.MSEn(ventana, Mobj, Scales=escalas)
        caracteristicas.append(entropias)
        
    return np.array(caracteristicas)

# --- Sección de prueba local ---
if __name__ == "__main__":
    ruta_prueba = "Tarea1/data/normal/97.mat" 
    senal_prueba = cargar_senal_cwru(ruta_prueba)
    ventanas_prueba = particionar_ventanas(senal_prueba, tamano_ventana=600)
    print(f"Forma de las ventanas: {ventanas_prueba.shape}")
    
    # Prueba matemática
    print("\nCalculando MPE para las 2 primeras ventanas...")
    mpe_resultados = calcular_mpe(ventanas_prueba[:2]) 
    print(f"Resultados MPE:\n{mpe_resultados}")

