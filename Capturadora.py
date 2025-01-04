import csv
import mindwave
import time
import datetime
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class RawDataBuffer:
    """Clase para manejar el buffer de datos raw"""
    def __init__(self):
        self.buffer = []
        self.timestamps = []
        self.total_samples = 0
    
    def add_data(self, value, timestamp):
        self.total_samples += 1
        self.buffer.append(value)
        self.timestamps.append(timestamp)
        return True
    
    def get_data(self):
        return self.buffer.copy(), self.timestamps.copy()
    
    def clear(self):
        self.buffer = []
        self.timestamps = []
        self.total_samples = 0

def initialize_plot():
    """Inicializa la configuración de las gráficas"""
    plt.style.use('default')
    
    # Crear figura principal
    fig = plt.figure(figsize=(15, 20))
    gs = plt.GridSpec(12, 4, figure=fig)
    
    # Crear subplots
    ax1 = fig.add_subplot(gs[0:2, :])
    ax2 = fig.add_subplot(gs[2:4, :])
    
    # Inicializar línea para datos raw
    line_raw, = ax1.plot([], [], 'b-', linewidth=0.5, label='Raw EEG')
    ax1.set_title('Señal Raw EEG')
    ax1.set_xlabel('Muestras')
    ax1.set_ylabel('Amplitud')
    ax1.grid(True)
    
    # Inicializar líneas para atención y meditación
    lines_att_med = {}
    att_med_colors = ['#FF4444', '#4444FF']  # Rojo para atención, Azul para meditación
    for signal, color in zip(['attention', 'meditation'], att_med_colors):
        lines_att_med[signal], = ax2.plot([], [], label=signal, color=color, linewidth=2)
    ax2.set_title('Señales de Atención y Meditación')
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Valor')
    ax2.legend()
    ax2.grid(True)
    
    # Crear subplots para cada onda cerebral
    wave_axes = []
    wave_lines = {}
    wave_signals = ['delta', 'theta', 'low-alpha', 'high-alpha', 
                    'low-beta', 'high-beta', 'low-gamma', 'mid-gamma']
    
    # Definir colores para cada onda
    wave_colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF',  # Rojo, Verde, Azul, Magenta
                  '#FFA500', '#800080', '#008080', '#FFD700']    # Naranja, Púrpura, Teal, Dorado
    
    row_positions = [4, 4, 6, 6, 8, 8, 10, 10]
    col_positions = [0, 2, 0, 2, 0, 2, 0, 2]
    
    for i, signal in enumerate(wave_signals):
        ax = fig.add_subplot(gs[row_positions[i]:row_positions[i]+2, 
                             col_positions[i]:col_positions[i]+2])
        wave_axes.append(ax)
        wave_lines[signal], = ax.plot([], [], label=signal, color=wave_colors[i], linewidth=1.5)
        ax.set_title(f'Onda {signal}')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Amplitud')
        ax.legend()
        ax.grid(True)
    
    fig.tight_layout(pad=3.0)
    return fig, ax1, ax2, wave_axes, line_raw, lines_att_med, wave_lines

def live_plot(dataWaves, raw_buffer):
    """Función para actualizar las gráficas en tiempo real"""
    global fig, ax1, ax2, wave_axes, line_raw, lines_att_med, wave_lines
    
    # Actualizar datos raw
    raw_data, _ = raw_buffer.get_data()
    if raw_data:
        data_to_plot = raw_data[-5120:]
        line_raw.set_data(range(len(data_to_plot)), data_to_plot)
        ax1.relim()
        ax1.autoscale_view()
    
    # Actualizar datos de atención y meditación
    time = np.arange(len(dataWaves['attention']))
    for signal in ['attention', 'meditation']:
        lines_att_med[signal].set_data(time, dataWaves[signal])
    ax2.relim()
    ax2.autoscale_view()
    ax2.set_ylim(-1, 105)
    
    # Actualizar ondas cerebrales
    for signal in wave_lines:
        wave_lines[signal].set_data(time, dataWaves[signal])
        ax = wave_lines[signal].axes
        ax.relim()
        ax.autoscale_view()
        if len(time) > 100:
            ax.set_xlim(len(time) - 100, len(time) + 10)
    
    fig.canvas.draw()
    fig.canvas.flush_events()

def save_final_plot(filename, dataWaves, raw_buffer):
    """Guardar la gráfica final como imagen"""
    fig_final = plt.figure(figsize=(15, 20))
    gs = plt.GridSpec(12, 4, figure=fig_final)
    
    # Graficar datos raw
    ax1_final = fig_final.add_subplot(gs[0:2, :])
    raw_data, _ = raw_buffer.get_data()
    if raw_data:
        ax1_final.plot(raw_data, 'b-', linewidth=0.5)
        ax1_final.set_title('Señal Raw EEG (Sesión Completa)')
        ax1_final.set_xlabel('Muestras')
        ax1_final.set_ylabel('Amplitud')
        ax1_final.grid(True)
    
    # Graficar atención y meditación
    ax2_final = fig_final.add_subplot(gs[2:4, :])
    time = np.arange(len(dataWaves['attention']))
    att_med_colors = ['#FF4444', '#4444FF']
    for signal, color in zip(['attention', 'meditation'], att_med_colors):
        ax2_final.plot(time, dataWaves[signal], label=signal, color=color, linewidth=2)
    ax2_final.set_title('Señales de Atención y Meditación')
    ax2_final.set_xlabel('Tiempo')
    ax2_final.set_ylabel('Valor')
    ax2_final.legend()
    ax2_final.grid(True)
    
    # Graficar ondas cerebrales individuales
    wave_signals = ['delta', 'theta', 'low-alpha', 'high-alpha', 
                    'low-beta', 'high-beta', 'low-gamma', 'mid-gamma']
    
    wave_colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF',
                  '#FFA500', '#800080', '#008080', '#FFD700']
    
    row_positions = [4, 4, 6, 6, 8, 8, 10, 10]
    col_positions = [0, 2, 0, 2, 0, 2, 0, 2]
    
    for i, signal in enumerate(wave_signals):
        ax = fig_final.add_subplot(gs[row_positions[i]:row_positions[i]+2, 
                                    col_positions[i]:col_positions[i]+2])
        ax.plot(time, dataWaves[signal], label=signal, color=wave_colors[i], linewidth=1.5)
        ax.set_title(f'Onda {signal}')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Amplitud')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    image_filename = os.path.splitext(os.path.basename(filename))[0] + '.png'
    image_path = os.path.join('Images', image_filename)
    
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close(fig_final)
    print(f'Gráfica guardada como: {image_path}')

def main():
    # Solicitud de datos
    usuario = input('Nombre de usuario: ')
    
    # Mostrar opciones de sesión
    print('Selecciona el tipo de sesión:')
    TipesSess = {
        '1': 'Leer',
        '2': 'Ver_video',
        '3': 'Jugar_Speed_Math',
        '4': 'Juego_de_carros'
    }
    
    for key, value in TipesSess.items():
        print(f'{key}: {value}')
    
    SessionTipeOpc = input('Ingresa el número correspondiente: ')
    sessionTipe = TipesSess.get(SessionTipeOpc, 'Leer')
    
    if sessionTipe == 'Leer' and SessionTipeOpc not in TipesSess:
        print('Opción no válida. Por defecto se seleccionará "Leer".')
    
    # Pedir duración de la sesión
    print('Especifica la duración de la sesión en minutos (o presiona Enter para usar el valor por defecto de 10 minutos):')
    TimeSess = input('Duración (minutos): ')
    
    try:
        TimeSeg = int(TimeSess) * 60 if TimeSess else 10 * 60
    except ValueError:
        print('Valor no válido, usando duración por defecto de 10 minutos.')
        TimeSeg = 10 * 60

    # Crear nombre de archivo con timestamp
    ts = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    filename = f'DataSet/{usuario}_{sessionTipe}_{ts}.dat'
    raw_filename = f'DataSet/raw_{usuario}_{sessionTipe}_{ts}.dat'
    print(f'Escribiendo en {filename} y {raw_filename}')

    # Conexión con mindwave
    print('Conectando al dispositivo MindWave...')
    headset = mindwave.Headset('/dev/ttyACM0')
    print('Conectado, esperando 10 segundos para que los datos comiencen a transmitirse...')
    time.sleep(10)

    # Inicializar el buffer para datos raw y el diccionario para otras señales
    raw_buffer = RawDataBuffer()
    dataWaves = collections.defaultdict(list)
    
    print('Comenzando a grabar...')
    start_time = time.time()
    last_stats_time = start_time
    
    with open(filename, 'w') as f, open(raw_filename, 'w') as raw_f:
        writer = csv.writer(f)
        raw_writer = csv.writer(raw_f)
        
        writer.writerow(['Timestamp', 'Attention', 'Meditation', 
                        'delta', 'theta', 'low-alpha', 'high-alpha',
                        'low-beta', 'high-beta', 'low-gamma', 'mid-gamma'])
        raw_writer.writerow(['Timestamp', 'Raw_Value'])
        
        end_time = time.time() + TimeSeg
        last_update = time.time()
        
        while time.time() < end_time:
            current_time = time.time()
            ts = datetime.datetime.now().isoformat()
            
            # Procesar datos raw
            current_raw = headset.raw_value
            raw_buffer.add_data(current_raw, current_time)
            raw_writer.writerow([ts, current_raw])
            
            # Mostrar estadísticas cada segundo
            if current_time - last_stats_time >= 1.0:
                rate = raw_buffer.total_samples / (current_time - start_time)
                print(f"\rMuestras: {raw_buffer.total_samples} | Tasa: {rate:.2f} Hz", end='')
                last_stats_time = current_time
            
            # Actualizar otras señales cada segundo
            if len(dataWaves['attention']) == 0 or current_time - last_update >= 1.0:
                last_update = current_time
                
                dataWaves['attention'].append(headset.attention)
                dataWaves['meditation'].append(headset.meditation)
                
                for wave_name in ['delta', 'theta', 'low-alpha', 'high-alpha',
                                'low-beta', 'high-beta', 'low-gamma', 'mid-gamma']:
                    dataWaves[wave_name].append(headset.waves.get(wave_name, 0))
                
                values = [ts, headset.attention, headset.meditation]
                values.extend(list(headset.waves.values()))
                writer.writerow(values)
                
                live_plot(dataWaves, raw_buffer)

            time.sleep(0.0009)
    
    # Estadísticas finales
    elapsed_time = time.time() - start_time
    print(f"\n\nEstadísticas finales:")
    print(f"Tiempo total: {elapsed_time:.2f} segundos")
    print(f"Total muestras: {raw_buffer.total_samples}")
    print(f"Tasa promedio: {raw_buffer.total_samples/elapsed_time:.2f} Hz")
    
    # Guardar gráfica final
    save_final_plot(filename, dataWaves, raw_buffer)
    plt.close('all')

if __name__ == "__main__":
    # Asegurarse de que existen las carpetas necesarias
    if not os.path.exists('DataSet'):
        os.makedirs('DataSet')
    if not os.path.exists('Images'):
        os.makedirs('Images')
        
    # Inicializar variables globales para la gráfica
    fig, ax1, ax2, wave_axes, line_raw, lines_att_med, wave_lines = initialize_plot()
    plt.ion()
    plt.show()
    
    # Ejecutar programa principal
    main()