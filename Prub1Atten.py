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
results_file = os.path.join(results_dir, f"ResultsATT_{usuario}.txt")
plot_filename = os.path.join(results_dir, f"GraficaATT_{usuario}.png")

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

print(f"\nAnálisis de atención para usuario: {usuario}")
print("="*50)

# Cargar datos
dataframes = {}
for nombre, actividad in actividades.items():
    try:
        dataframes[nombre] = load_eeg_csv(usuario, actividad)
    except FileNotFoundError as e:
        print(e)

# Análisis estadístico
stats_data = []
for nombre, df in dataframes.items():
    stats_dict = {
        'Actividad': nombre,
        'Media': df['Attention'].mean(),
        'Mediana': df['Attention'].median(),
        'Std': df['Attention'].std(),
        'Min': df['Attention'].min(),
        'Max': df['Attention'].max(),
        '25%': df['Attention'].quantile(0.25),
        '75%': df['Attention'].quantile(0.75)
    }
    stats_data.append(stats_dict)

df_stats = pd.DataFrame(stats_data)
df_stats_sorted = df_stats.sort_values(by='Media', ascending=False)

# Configurar estilo de las gráficas
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Crear figura con subplots
fig = plt.figure(figsize=(15, 10))

# 1. Gráfico de barras de medias con barras de error
ax1 = plt.subplot(2, 2, 1)
sns.barplot(data=df_stats_sorted, x='Actividad', y='Media', ax=ax1,
            palette='viridis')
ax1.set_title('Media de Atención por Actividad')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

# 2. Box Plot usando seaborn
ax2 = plt.subplot(2, 2, 2)
plot_data = pd.DataFrame({
    'Actividad': [act for act in df_stats_sorted['Actividad'] for _ in range(len(dataframes[act]['Attention']))],
    'Attention': pd.concat([dataframes[act]['Attention'] for act in df_stats_sorted['Actividad']])
})
sns.boxplot(data=plot_data, x='Actividad', y='Attention', ax=ax2, palette='viridis')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.set_title('Distribución de Atención por Actividad')

# 3. Líneas de tiempo superpuestas
ax3 = plt.subplot(2, 1, 2)
for nombre in df_stats_sorted['Actividad']:
    df = dataframes[nombre]
    tiempo_minutos = np.arange(len(df)) / 60
    ax3.plot(tiempo_minutos, df['Attention'], label=nombre, alpha=0.7)
ax3.set_title('Evolución de la Atención durante la Sesión')
ax3.set_xlabel('Tiempo (minutos)')
ax3.set_ylabel('Nivel de Atención')
ax3.legend()

plt.tight_layout()

# Mostrar y guardar gráfica
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Imprimir estadísticas
print("\nEstadísticas descriptivas de atención por actividad:")
print("="*50)
print(df_stats_sorted[['Actividad', 'Media', 'Mediana', 'Min', 'Max', 'Std']].round(2).to_string(index=False))

# ANOVA y explicación
attention_groups = [df['Attention'] for df in dataframes.values()]
f_statistic, p_value = stats.f_oneway(*attention_groups)
print("\nAnálisis de diferencias entre actividades (ANOVA):")
print("="*50)
print(f"F-statistic: {f_statistic:.4f}")
print(f"Valor p: {p_value:.4f}")
if p_value < 0.05:
    print("Conclusión: Hay diferencias significativas en la atención entre las actividades")
else:
    print("Conclusión: No hay diferencias significativas en la atención entre las actividades")

# Encontrar la actividad con mejor atención
mejor_actividad = df_stats_sorted.iloc[0]
print(f"\nResumen final:")
print("="*50)
print(f"La actividad con mejor atención promedio fue: {mejor_actividad['Actividad']}")
print(f"Con una media de: {mejor_actividad['Media']:.2f}")
print(f"\nResultados guardados en:")
print(f"- Texto: {results_file}")
print(f"- Gráficas: {plot_filename}")

# Restaurar stdout
sys.stdout = sys.stdout.terminal