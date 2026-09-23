import scipy.io
import numpy as np
from scipy.stats import norm

def cargar_senal_cwru(ruta_archivo):
    """Carga un archivo .mat del dataset CWRU y extrae la señal DE_time."""
    mat_data = scipy.io.loadmat(ruta_archivo)
    claves_de = [key for key in mat_data.keys() if 'DE_time' in key]
    
    if not claves_de:
        raise ValueError("No se encontró la variable DE_time en el archivo.")
        
    senal = mat_data[claves_de[0]].flatten()
    return senal

def particionar_ventanas(senal, tamano_ventana):
    """Particiona una señal 1D en múltiples ventanas de tamaño W."""
    num_ventanas = len(senal) // tamano_ventana
    senal_recortada = senal[:num_ventanas * tamano_ventana]
    ventanas = np.reshape(senal_recortada, (num_ventanas, tamano_ventana))
    return ventanas

def coarse_graining(senal, escala):
    """Aplica el coarse-graining promediando la señal en bloques del tamaño de la escala."""
    if escala == 1:
        return senal
    n = len(senal)
    m = n // escala
    return np.mean(senal[:m * escala].reshape(m, escala), axis=1)

def permutation_entropy(senal, m, tau):
    """Calcula la Entropía de Permutación estándar para una escala dada."""
    n = len(senal)
    if n < m * tau:
        return 0.0
    
    patrones = []
    for i in range(n - (m - 1) * tau):
        patron = tuple(np.argsort(senal[i : i + m*tau : tau]))
        patrones.append(patron)
        
    _, counts = np.unique(patrones, axis=0, return_counts=True)
    p = counts / len(patrones)
    
    return -np.sum(p * np.log(p + 1e-10))

def dispersion_entropy(senal, m, tau, c):
    """Calcula la Entropía de Dispersión estándar para una escala dada."""
    n = len(senal)
    if n < m * tau:
        return 0.0
    
    mu, sigma = np.mean(senal), np.std(senal)
    if sigma == 0:
        return 0.0
    y = norm.cdf(senal, loc=mu, scale=sigma)
    
    z = np.round(c * y + 0.5).astype(int)
    z = np.clip(z, 1, c)
    
    patrones = []
    for i in range(n - (m - 1) * tau):
        patron = tuple(z[i : i + m*tau : tau])
        patrones.append(patron)
        
    _, counts = np.unique(patrones, axis=0, return_counts=True)
    p = counts / len(patrones)
    
    return -np.sum(p * np.log(p + 1e-10))

def calcular_mpe(ventanas, m=3, tau=1, escalas=5):
    """Calcula la Entropía Multiescala de Permutación (MPE)."""
    caracteristicas = []
    for ventana in ventanas:
        entropias = []
        for s in range(1, escalas + 1):
            cg_senal = coarse_graining(ventana, s)
            entropias.append(permutation_entropy(cg_senal, m, tau))
        caracteristicas.append(entropias)
    return np.array(caracteristicas)

def calcular_mde(ventanas, m=2, tau=1, c=3, escalas=5):
    """Calcula la Entropía Multiescala de Dispersión (MDE)."""
    caracteristicas = []
    for ventana in ventanas:
        entropias = []
        for s in range(1, escalas + 1):
            cg_senal = coarse_graining(ventana, s)
            entropias.append(dispersion_entropy(cg_senal, m, tau, c))
        caracteristicas.append(entropias)
    return np.array(caracteristicas)

if __name__ == "__main__":
    print("1. Cargando señal...")
    ruta_prueba = "data/fallos/108.mat"
    senal_prueba = cargar_senal_cwru(ruta_prueba)
    ventanas_prueba = particionar_ventanas(senal_prueba, tamano_ventana=600)
    
    print("\n2. Prueba matemática MPE (Manual)...")
    mpe_resultados = calcular_mpe(ventanas_prueba[:2]) 
    print(f"Resultados MPE:\n{mpe_resultados}")
    
    print("\n3. Prueba matemática MDE (Manual)...")
    mde_resultados = calcular_mde(ventanas_prueba[:2]) 
    print(f"Resultados MDE:\n{mde_resultados}")