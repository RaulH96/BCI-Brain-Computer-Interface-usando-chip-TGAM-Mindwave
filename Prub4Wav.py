import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
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

results_file = os.path.join(results_dir, f"ResultsWav_{usuario}.txt")
plot_filename = os.path.join(results_dir, f"GraficaWav_{usuario}.png")

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(results_file)

def load_raw_eeg_csv(usuario, actividad):
    pattern = f"DataSet/raw_{usuario}_{actividad}_*.dat"
    file_candidates = glob.glob(pattern)
    if not file_candidates:
        raise FileNotFoundError(f"No se encontró archivo raw para {usuario} - {actividad}")
    
    file_to_load = file_candidates[0]
    print(f"Cargando archivo raw: {file_to_load}")
    return pd.read_csv(file_to_load)

def calculate_wavelet(signal_data, fs=100):
    """Calcula la transformada wavelet usando scipy.signal"""
    widths = np.arange(1, 128)
    frequencies = fs * (1 / widths)
    
    # Crear un wavelet para cada escala
    wavelets = [signal.morlet2(M=int(w * 10), s=w, w=5.0) for w in widths]
    cwtmatr = np.array([signal.convolve(signal_data, wav, mode='same') for wav in wavelets])
    
    return cwtmatr, frequencies
# Procesar y graficar
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.ravel()

for idx, (nombre, actividad) in enumerate(actividades.items()):
    try:
        # Cargar datos raw
        df_raw = load_raw_eeg_csv(usuario, actividad)
        raw_signal = df_raw['Raw_Value'].values
        
        # Tomar una muestra de 10 segundos (1000 puntos)
        sample_size = 1000
        if len(raw_signal) > sample_size:
            start_idx = len(raw_signal) // 2
            raw_signal = raw_signal[start_idx:start_idx+sample_size]
        
        # Normalizar señal
        raw_signal = (raw_signal - np.mean(raw_signal)) / np.std(raw_signal)
        
        # Calcular CWT
        coefficients, frequencies = calculate_wavelet(raw_signal)
        
        # Graficar
        ax = axes[idx]
        times = np.arange(len(raw_signal))/100  # Tiempo en segundos
        im = ax.pcolormesh(times, frequencies, np.abs(coefficients), 
                          shading='gouraud', cmap='jet')
        
        # Añadir líneas para las bandas
        for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
            ax.axhline(y=low_freq, color='w', linestyle='--', alpha=0.5)
            ax.text(0, low_freq, band_name, color='white', 
                   backgroundcolor='black', alpha=0.7)
        
        ax.set_title(f'Análisis Wavelet - {nombre}')
        ax.set_ylabel('Frecuencia (Hz)')
        ax.set_xlabel('Tiempo (s)')
        plt.colorbar(im, ax=ax, label='Magnitud')
        
        # Ajustar escala de frecuencia
        ax.set_yscale('log')
        ax.set_ylim(1, 50)
        
        print(f"\nAnálisis Wavelet para {nombre}:")
        # Calcular energía por banda
        for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
            mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            band_energy = np.mean(np.abs(coefficients[mask, :]))
            print(f"Energía en banda {band_name}: {band_energy:.2f}")
        
    except Exception as e:
        print(f"Error procesando {nombre}: {str(e)}")

plt.tight_layout()
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Restaurar stdout
sys.stdout = sys.stdout.terminal