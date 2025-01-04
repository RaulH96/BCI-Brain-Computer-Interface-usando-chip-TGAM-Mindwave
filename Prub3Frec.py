import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import scipy.fft as fft
import os
import sys
import glob

# Usuario y actividades
usuario = "2015170726"
actividades = {
    'Leer': 'Leer',
    'Ver_video': 'Ver_video',
    'Speed_Math': 'Jugar_Speed_Math',
    'Carros': 'Juego_de_carros'
}

# Definir bandas de frecuencia (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

# Crear estructura de carpetas
results_dir = os.path.join('Results', usuario)
os.makedirs(results_dir, exist_ok=True)

def load_raw_eeg_csv(usuario, actividad):
    pattern = f"DataSet/raw_{usuario}_{actividad}_*.dat"
    file_candidates = glob.glob(pattern)
    if not file_candidates:
        raise FileNotFoundError(f"No se encontró archivo raw para {usuario} - {actividad}")
    
    file_to_load = file_candidates[0]
    print(f"Cargando archivo raw: {file_to_load}")
    return pd.read_csv(file_to_load)

def calculate_fft(raw_signal):
    """Calcula la FFT de la señal"""
    n = len(raw_signal)
    fft_result = fft.fft(raw_signal)
    # Usar frecuencia aproximada de 100 Hz para la visualización
    freqs = fft.fftfreq(n, 1/100)
    
    # Solo mantenemos la parte positiva del espectro
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    power = np.abs(fft_result[pos_mask])**2
    
    return freqs, power

# Procesar y graficar
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.ravel()

for idx, (nombre, actividad) in enumerate(actividades.items()):
    try:
        # Cargar datos raw
        df_raw = load_raw_eeg_csv(usuario, actividad)
        raw_signal = df_raw['Raw_Value'].values
        
        # Calcular FFT
        freqs, power = calculate_fft(raw_signal)
        
        # Graficar
        ax = axes[idx]
        ax.semilogy(freqs, power)
        ax.set_title(f'Espectro de Frecuencias - {nombre}')
        ax.set_xlabel('Frecuencia (Hz)')
        ax.set_ylabel('Potencia')
        ax.grid(True)
        
        # Colorear bandas de frecuencia
        for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
            ax.axvspan(low_freq, high_freq, alpha=0.2, label=band_name)
        ax.legend()
        
        # Limitar visualización hasta 100 Hz
        ax.set_xlim(0, 60)
        
    except Exception as e:
        print(f"Error procesando {nombre}: {str(e)}")

plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"GraficaFrec_{usuario}.png"), dpi=300)
plt.show()
plt.close()