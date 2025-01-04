
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QComboBox, 
                           QLineEdit, QGridLayout, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import mindwave
import time
import datetime
import collections
import csv
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

class MindwaveGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mindwave Monitor")
        self.setGeometry(100, 100, 1500, 1000)
        
        # Crear widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Panel de control
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Panel de gráficas
        self.fig = plt.figure(figsize=(15, 20))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        # Inicializar gráficas
        self.setup_plots()
        
        # Variables de estado
        self.recording = False
        self.headset = None
        self.raw_buffer = None
        self.dataWaves = None
        self.start_time = None
        
        # Timer para actualización de estadísticas
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.setInterval(1000)  # Actualizar cada segundo
        
        # IMPORTANTE: Este timer reemplaza al time.sleep(0.001)
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.capture_data)
        self.capture_timer.setInterval(1)  # 1ms = 0.001 segundos  <<------Aqui
        
    def create_control_panel(self):
        control_group = QGroupBox("Panel de Control")
        layout = QGridLayout()
        
        # Usuario
        layout.addWidget(QLabel("Usuario:"), 0, 0)
        self.user_input = QLineEdit()
        layout.addWidget(self.user_input, 0, 1)
        
        # Tipo de sesión
        layout.addWidget(QLabel("Tipo de Sesión:"), 0, 2)
        self.session_combo = QComboBox()
        self.session_combo.addItems(['Leer', 'Ver_video', 'Jugar_Speed_Math', 'Juego_de_carros'])
        layout.addWidget(self.session_combo, 0, 3)
        
        # Duración
        layout.addWidget(QLabel("Duración (min):"), 0, 4)
        self.duration_input = QLineEdit("10")
        layout.addWidget(self.duration_input, 0, 5)
        
        # Botones
        self.start_button = QPushButton("Iniciar")
        self.start_button.clicked.connect(self.toggle_recording)
        layout.addWidget(self.start_button, 0, 6)
        
        # Estadísticas
        self.stats_label = QLabel("Esperando inicio...")
        layout.addWidget(self.stats_label, 1, 0, 1, 7)
        
        control_group.setLayout(layout)
        return control_group
    
    def setup_plots(self):
        gs = self.fig.add_gridspec(12, 4)
        
        # Raw EEG
        self.ax1 = self.fig.add_subplot(gs[0:2, :])
        self.line_raw, = self.ax1.plot([], [], 'b-', linewidth=0.5, label='Raw EEG')
        self.ax1.set_title('Señal Raw EEG')
        self.ax1.set_xlabel('Muestras')
        self.ax1.set_ylabel('Amplitud')
        self.ax1.grid(True)
        
        # Atención/Meditación
        self.ax2 = self.fig.add_subplot(gs[2:4, :])
        self.lines_att_med = {}
        att_med_colors = ['#FF4444', '#4444FF']  # Rojo para atención, Azul para meditación
        for signal, color in zip(['attention', 'meditation'], att_med_colors):
            self.lines_att_med[signal], = self.ax2.plot([], [], label=signal, color=color, linewidth=2)
        self.ax2.set_title('Señales de Atención y Meditación')
        self.ax2.set_xlabel('Tiempo')
        self.ax2.set_ylabel('Valor')
        self.ax2.legend()
        self.ax2.grid(True)

        # Ondas cerebrales
        self.wave_lines = {}
        wave_signals = ['delta', 'theta', 'low-alpha', 'high-alpha', 
                       'low-beta', 'high-beta', 'low-gamma', 'mid-gamma']
        wave_colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF',
                      '#FFA500', '#800080', '#008080', '#FFD700']
        
        row_positions = [4, 4, 6, 6, 8, 8, 10, 10]
        col_positions = [0, 2, 0, 2, 0, 2, 0, 2]
        
        self.wave_axes = []
        for i, signal in enumerate(wave_signals):
            ax = self.fig.add_subplot(gs[row_positions[i]:row_positions[i]+2, 
                                      col_positions[i]:col_positions[i]+2])
            self.wave_axes.append(ax)
            self.wave_lines[signal], = ax.plot([], [], label=signal, color=wave_colors[i], linewidth=1.5)
            ax.set_title(f'Onda {signal}')
            ax.set_xlabel('Tiempo')
            ax.set_ylabel('Amplitud')
            ax.legend()
            ax.grid(True)
        
        self.fig.tight_layout(pad=3.0)

    def capture_data(self):
        """Función que reemplaza el bucle principal con time.sleep(0.001)"""
        if not self.recording:
            return
            
        current_time = time.time()
        
        # Verificar si terminó el tiempo
        if current_time >= self.end_time:
            self.stop_recording()
            return
        
        try:
            # Capturar y guardar dato raw
            ts = datetime.datetime.now().isoformat()
            current_raw = self.headset.raw_value
            self.raw_buffer.add_data(current_raw, current_time)
            self.raw_writer.writerow([ts, current_raw])
            
            # Actualizar otras señales cada segundo
            if current_time - self.last_update >= 1.0:
                self.update_signals(ts, current_time)
                self.last_update = current_time
            
        except Exception as e:
            print(f"Error en captura: {e}")
    
    def update_signals(self, ts, current_time):
        """Actualización de señales y gráficas (cada segundo)"""
        # Actualizar señales
        self.dataWaves['attention'].append(self.headset.attention)
        self.dataWaves['meditation'].append(self.headset.meditation)
        
        for wave_name in ['delta', 'theta', 'low-alpha', 'high-alpha',
                         'low-beta', 'high-beta', 'low-gamma', 'mid-gamma']:
            self.dataWaves[wave_name].append(self.headset.waves.get(wave_name, 0))
        
        # Guardar datos
        values = [ts, self.headset.attention, self.headset.meditation]
        values.extend(list(self.headset.waves.values()))
        self.writer.writerow(values)
        
        # Actualizar gráficas
        self.update_plots()
    
    def update_plots(self):
        # Actualizar raw
        raw_data, _ = self.raw_buffer.get_data()
        if raw_data:
            data_to_plot = raw_data[-5120:]
            self.line_raw.set_data(range(len(data_to_plot)), data_to_plot)
            self.ax1.relim()
            self.ax1.autoscale_view()
        
        # Actualizar atención/meditación
        time = np.arange(len(self.dataWaves['attention']))
        for signal in ['attention', 'meditation']:
            self.lines_att_med[signal].set_data(time, self.dataWaves[signal])
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2.set_ylim(-1, 105)
        
        # Actualizar ondas
        for signal in self.wave_lines:
            self.wave_lines[signal].set_data(time, self.dataWaves[signal])
            ax = self.wave_lines[signal].axes
            ax.relim()
            ax.autoscale_view()
            if len(time) > 100:
                ax.set_xlim(len(time) - 100, len(time) + 10)
        
        self.canvas.draw()

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        try:
            # Validar entradas
            if not self.user_input.text():
                raise ValueError("Por favor ingrese un usuario")
            
            duration = int(self.duration_input.text()) * 60
            if duration <= 0:
                raise ValueError("La duración debe ser mayor a 0")
            
            # Configurar grabación
            self.setup_recording(duration)
            
            # Iniciar grabación
            self.recording = True
            self.start_button.setText("Detener")
            self.stats_timer.start()
            self.capture_timer.start()  # Equivalente al bucle con time.sleep(0.001)
            
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", str(e))
    
    def setup_recording(self, duration):
        # Crear archivos
        ts = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        user = self.user_input.text()
        session = self.session_combo.currentText()
        
        os.makedirs('DataSet', exist_ok=True)
        os.makedirs('Images', exist_ok=True)
        
        self.filename = f'DataSet/{user}_{session}_{ts}.dat'
        self.raw_filename = f'DataSet/raw_{user}_{session}_{ts}.dat'
        
        # Conectar al dispositivo
        print('Conectando al dispositivo MindWave...')
        self.headset = mindwave.Headset('/dev/ttyACM0')
        print('Conectado, esperando 10 segundos para que los datos comiencen a transmitirse...')
        time.sleep(10)
        
        # Inicializar buffers
        self.raw_buffer = RawDataBuffer()
        self.dataWaves = collections.defaultdict(list)
        
        # Configurar archivos
        self.f = open(self.filename, 'w', newline='')
        self.raw_f = open(self.raw_filename, 'w', newline='')
        
        self.writer = csv.writer(self.f)
        self.raw_writer = csv.writer(self.raw_f)
        
        self.writer.writerow(['Timestamp', 'Attention', 'Meditation', 
                            'delta', 'theta', 'low-alpha', 'high-alpha',
                            'low-beta', 'high-beta', 'low-gamma', 'mid-gamma'])
        self.raw_writer.writerow(['Timestamp', 'Raw_Value'])
        
        self.start_time = time.time()
        self.end_time = self.start_time + duration
        self.last_update = self.start_time

    def update_stats(self):
        """Actualización de estadísticas (cada segundo)"""
        if not self.recording:
            return
            
        current_time = time.time()
        rate = self.raw_buffer.total_samples / (current_time - self.start_time)
        
        self.stats_label.setText(
            f"Muestras: {self.raw_buffer.total_samples} | "
            f"Tasa: {rate:.2f} Hz | "
            f"Tiempo restante: {int(self.end_time - current_time)}s"
        )
    
    def save_final_plot(self):
        if not self.filename:
            return
            
        image_filename = os.path.splitext(os.path.basename(self.filename))[0] + '.png'
        image_path = os.path.join('Images', image_filename)
        
        self.fig.savefig(image_path, dpi=300, bbox_inches='tight')
        print(f'Gráfica guardada como: {image_path}')
    
    def stop_recording(self):
        self.recording = False
        self.start_button.setText("Iniciar")
        self.stats_timer.stop()
        self.capture_timer.stop()
        
        if hasattr(self, 'f') and self.f:
            self.f.close()
        if hasattr(self, 'raw_f') and self.raw_f:
            self.raw_f.close()
        
        if self.headset:
            self.headset.stop()
        
        # Guardar gráfica final
        self.save_final_plot()
        
        # Mostrar estadísticas finales
        elapsed_time = time.time() - self.start_time
        stats = (
            f"Estadísticas finales:\n"
            f"Tiempo total: {elapsed_time:.2f} segundos\n"
            f"Total muestras: {self.raw_buffer.total_samples}\n"
            f"Tasa promedio: {self.raw_buffer.total_samples/elapsed_time:.2f} Hz"
        )
        self.stats_label.setText(stats)

    def closeEvent(self, event):
        if self.recording:
            self.stop_recording()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MindwaveGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()