import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os
from datetime import datetime
import sys

# Usuario y actividades
usuario = "2015170726"
actividades = {
    'Leer': 'Leer',
    'Ver_video': 'Ver_video',
    'Speed_Math': 'Jugar_Speed_Math',
    'Carros': 'Juego_de_carros'
}

# Crear estructura de carpetas
results_dir = os.path.join('Results', usuario)
os.makedirs(results_dir, exist_ok=True)

# Definir nombres de archivos
results_file = os.path.join(results_dir, f"ResultsCORR_{usuario}.txt")
plot_filename = os.path.join(results_dir, f"GraficaCORR_{usuario}.png")

# Redirigir stdout a archivo y terminal
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

def load_eeg_csv(usuario, actividad):
    pattern = f"DataSet/{usuario}_{actividad}_*.dat"
    file_candidates = glob.glob(pattern)
    if not file_candidates:
        raise FileNotFoundError(f"No se encontró archivo para {usuario} - {actividad}")
    
    file_to_load = file_candidates[0]
    print(f"Cargando archivo: {file_to_load}")
    return pd.read_csv(file_to_load)

print(f"\nAnálisis de correlación para usuario: {usuario}")
print("="*50)

# Cargar datos
dataframes = {}
for nombre, actividad in actividades.items():
    try:
        df = load_eeg_csv(usuario, actividad)
        df['Tarea'] = nombre
        dataframes[nombre] = df
    except FileNotFoundError as e:
        print(e)

# Unir todos los dataframes
df_all = pd.concat(dataframes.values(), ignore_index=True)

# Definir columnas para análisis
wave_columns = ['delta', 'theta', 'low-alpha', 'high-alpha',
                'low-beta', 'high-beta', 'low-gamma', 'mid-gamma']
analysis_columns = ['Attention'] + wave_columns

# Crear figura para todas las visualizaciones
plt.style.use('default')
fig = plt.figure(figsize=(20, 15))

# 1. Matriz de correlación general
print("\nMatriz de correlación general:")
print("="*50)
correlation_matrix = df_all[analysis_columns].corr()
print(correlation_matrix.round(3))

# Visualizar matriz de correlación
ax1 = plt.subplot(2, 2, 1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
ax1.set_title('Matriz de Correlación General')

# 2. Correlaciones con Atención por onda
correlations = {}
p_values = {}
for wave in wave_columns:
    corr, p_value = stats.pearsonr(df_all['Attention'], df_all[wave])
    correlations[wave] = corr
    p_values[wave] = p_value

# Visualizar correlaciones con Atención
ax2 = plt.subplot(2, 2, 2)
correlation_df = pd.DataFrame(list(correlations.items()), 
                            columns=['Onda', 'Correlación'])
sns.barplot(data=correlation_df, x='Onda', y='Correlación', ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.set_title('Correlación entre Ondas y Atención')

# 3. Análisis por tarea
print("\nCorrelaciones por tarea:")
print("="*50)
task_correlations = {}
for tarea, df in dataframes.items():
    task_correlations[tarea] = {}
    print(f"\nTarea: {tarea}")
    for wave in wave_columns:
        corr, p_value = stats.pearsonr(df['Attention'], df[wave])
        task_correlations[tarea][wave] = corr
        significance = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"{wave}: {corr:.3f} {significance}")

# Visualizar correlaciones por tarea
ax3 = plt.subplot(2, 1, 2)
task_corr_df = pd.DataFrame(task_correlations).T
sns.heatmap(task_corr_df, annot=True, cmap='coolwarm', center=0, ax=ax3)
ax3.set_title('Correlaciones por Tarea')

plt.tight_layout()

# Guardar y mostrar gráfica
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print(f"\nResultados guardados en:")
print(f"- Texto: {results_file}")
print(f"- Gráficas: {plot_filename}")

# Restaurar stdout
sys.stdout = sys.stdout.terminal

if __name__ == '__main__':
    main()