import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QComboBox, 
                           QLineEdit, QTabWidget, QMenuBar, QAction, QMessageBox,
                           QGridLayout, QGroupBox, QSizePolicy)
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import signal
from scipy import stats
import os
import glob
from datetime import datetime

class EEGAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analizador EEG")
        self.setGeometry(100, 100, 1200, 800)
        
        # Crear menú
        self.createMenu()
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Panel de control superior
        control_panel = self.createControlPanel()
        layout.addWidget(control_panel)
        
        # Crear pestañas
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Inicializar pestañas
        self.initializeTabs()
        
        # Variables de estado
        self.dataframes = {}
        self.raw_dataframes = {}
        
    def createMenu(self):
        menubar = self.menuBar()
        viewMenu = menubar.addMenu('Ver')
        
        # Acciones para mostrar/ocultar pestañas
        self.test_actions = {}
        tests = [
            ("Atención", "tab_attention"),
            ("Correlación", "tab_correlation"),
            ("FFT", "tab_fft"),
            ("Wavelet", "tab_wavelet")
        ]
        
        for name, attr in tests:
            action = QAction(name, self, checkable=True)
            action.setChecked(True)
            action.triggered.connect(
                lambda checked, a=attr: self.toggleTab(a, checked))
            viewMenu.addAction(action)
            self.test_actions[name] = action
    
    def createControlPanel(self):
        group = QGroupBox("Control")
        layout = QHBoxLayout()
        
        # ID de Usuario
        user_layout = QHBoxLayout()
        user_layout.addWidget(QLabel("ID Usuario:"))
        self.user_input = QLineEdit()
        user_layout.addWidget(self.user_input)
        layout.addLayout(user_layout)
        
        # Botón de análisis
        self.analyze_button = QPushButton("Analizar")
        self.analyze_button.clicked.connect(self.runAnalysis)
        layout.addWidget(self.analyze_button)
        
        group.setLayout(layout)
        return group
    
    def initializeTabs(self):
        # Pestaña de Atención
        self.tab_attention = QWidget()
        self.tabs.addTab(self.tab_attention, "Atención")
        layout_att = QVBoxLayout(self.tab_attention)
        self.fig_attention = plt.figure(figsize=(10, 8))
        self.canvas_attention = FigureCanvas(self.fig_attention)
        layout_att.addWidget(self.canvas_attention)
        
        # Pestaña de Correlación
        self.tab_correlation = QWidget()
        self.tabs.addTab(self.tab_correlation, "Correlación")
        layout_corr = QVBoxLayout(self.tab_correlation)
        self.fig_correlation = plt.figure(figsize=(10, 8))
        self.canvas_correlation = FigureCanvas(self.fig_correlation)
        layout_corr.addWidget(self.canvas_correlation)
        
        # Pestaña de FFT
        self.tab_fft = QWidget()
        self.tabs.addTab(self.tab_fft, "FFT")
        layout_fft = QVBoxLayout(self.tab_fft)
        self.fig_fft = plt.figure(figsize=(10, 8))
        self.canvas_fft = FigureCanvas(self.fig_fft)
        layout_fft.addWidget(self.canvas_fft)
        
        # Pestaña de Wavelet
        self.tab_wavelet = QWidget()
        self.tabs.addTab(self.tab_wavelet, "Wavelet")
        layout_wav = QVBoxLayout(self.tab_wavelet)
        self.fig_wavelet = plt.figure(figsize=(10, 8))
        self.canvas_wavelet = FigureCanvas(self.fig_wavelet)
        layout_wav.addWidget(self.canvas_wavelet)
    
    def toggleTab(self, tab_name, checked):
        tab = getattr(self, tab_name)
        if checked:
            self.tabs.addTab(tab, tab_name.split('_')[1].capitalize())
        else:
            self.tabs.removeTab(self.tabs.indexOf(tab))
    
    def runAnalysis(self):
        usuario = self.user_input.text()
        if not usuario:
            QMessageBox.warning(self, "Error", "Por favor ingrese un ID de usuario")
            return
        
        try:
            # Crear directorio de resultados
            results_dir = os.path.join('Results', usuario)
            os.makedirs(results_dir, exist_ok=True)
            
            # Cargar datos
            self.loadData(usuario)
            
            # Ejecutar análisis
            if self.test_actions["Atención"].isChecked():
                self.runAttentionAnalysis(usuario)
            if self.test_actions["Correlación"].isChecked():
                self.runCorrelationAnalysis(usuario)
            if self.test_actions["FFT"].isChecked():
                self.runFFTAnalysis(usuario)
            if self.test_actions["Wavelet"].isChecked():
                self.runWaveletAnalysis(usuario)
            
            QMessageBox.information(self, "Éxito", "Análisis completado.\nResultados guardados en: {}".format(results_dir))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error durante el análisis: {str(e)}")
    
    def loadData(self, usuario):
        # Definir actividades
        actividades = {
            'Leer': 'Leer',
            'Ver_video': 'Ver_video',
            'Speed_Math': 'Jugar_Speed_Math',
            'Carros': 'Juego_de_carros'
        }
        
        # Cargar datos normales
        self.dataframes = {}
        for nombre, actividad in actividades.items():
            pattern = f"DataSet/{usuario}_{actividad}_*.dat"
            file_candidates = glob.glob(pattern)
            if file_candidates:
                self.dataframes[nombre] = pd.read_csv(file_candidates[0])
        
        # Cargar datos raw
        self.raw_dataframes = {}
        for nombre, actividad in actividades.items():
            pattern = f"DataSet/raw_{usuario}_{actividad}_*.dat"
            file_candidates = glob.glob(pattern)
            if file_candidates:
                self.raw_dataframes[nombre] = pd.read_csv(file_candidates[0])
                
    def runAttentionAnalysis(self, usuario):
    #"""Ejecuta el análisis de atención y actualiza la pestaña correspondiente"""
    # Configurar archivo de resultados
        results_file = os.path.join('Results', usuario, "ResultsATT_{}.txt".format(usuario))
        plot_filename = os.path.join('Results', usuario, "GraficaATT_{}.png".format(usuario))
    
        with open(results_file, 'w') as f:
        # Análisis estadístico
            stats_data = []
            for nombre, df in self.dataframes.items():
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
                f.write("Estadísticas para {}:\n".format(nombre))
                for key, value in stats_dict.items():
                     if key != 'Actividad':
                        f.write("{}: {:.2f}\n".format(key, value))
                     else:
                        f.write("{}: {}\n".format(key, value))
                f.write("\n")

            df_stats = pd.DataFrame(stats_data)
            df_stats_sorted = df_stats.sort_values(by='Media', ascending=False)

            # Limpiar figura anterior
            self.fig_attention.clear()

            # 1. Gráfico de barras de medias
            ax1 = self.fig_attention.add_subplot(221)
            sns.barplot(data=df_stats_sorted, x='Actividad', y='Media', ax=ax1,
                      palette='viridis')
            ax1.set_title('Media de Atención por Actividad')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

            # 2. Box Plot
            ax2 = self.fig_attention.add_subplot(222)
            plot_data = pd.DataFrame({
                'Actividad': [act for act in df_stats_sorted['Actividad'] 
                            for _ in range(len(self.dataframes[act]['Attention']))],
                'Attention': pd.concat([self.dataframes[act]['Attention'] 
                                     for act in df_stats_sorted['Actividad']])
            })
            sns.boxplot(data=plot_data, x='Actividad', y='Attention', 
                       ax=ax2, palette='viridis')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            ax2.set_title('Distribución de Atención por Actividad')

            # 3. Líneas de tiempo
            ax3 = self.fig_attention.add_subplot(212)
            for nombre in df_stats_sorted['Actividad']:
                df = self.dataframes[nombre]
                tiempo_minutos = np.arange(len(df)) / 60
                ax3.plot(tiempo_minutos, df['Attention'], label=nombre, alpha=0.7)
            ax3.set_title('Evolución de la Atención durante la Sesión')
            ax3.set_xlabel('Tiempo (minutos)')
            ax3.set_ylabel('Nivel de Atención')
            ax3.legend()

            # ANOVA
            attention_groups = [df['Attention'] for df in self.dataframes.values()]
            f_statistic, p_value = stats.f_oneway(*attention_groups)
            f.write("\nResultados ANOVA:\n")
            f.write(f"F-statistic: {f_statistic:.4f}\n")
            f.write(f"p-value: {p_value:.4f}\n")

            self.fig_attention.tight_layout()
            self.canvas_attention.draw()
            self.fig_attention.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    def runCorrelationAnalysis(self, usuario):
        """Ejecuta el análisis de correlación y actualiza la pestaña correspondiente"""
        # Configurar archivos de resultados
        results_file = os.path.join('Results', usuario, f"ResultsCORR_{usuario}.txt")
        plot_filename = os.path.join('Results', usuario, f"GraficaCORR_{usuario}.png")
        
        with open(results_file, 'w') as f:
            # Unir todos los dataframes
            df_all = pd.concat(self.dataframes.values(), ignore_index=True)
            
            # Definir columnas para análisis
            wave_columns = ['delta', 'theta', 'low-alpha', 'high-alpha',
                          'low-beta', 'high-beta', 'low-gamma', 'mid-gamma']
            analysis_columns = ['Attention'] + wave_columns
            
            # Matriz de correlación general
            f.write("\nMatriz de correlación general:\n")
            f.write("="*50 + "\n")
            correlation_matrix = df_all[analysis_columns].corr()
            f.write(correlation_matrix.round(3).to_string())
            f.write("\n\n")
            
            # Limpiar figura anterior
            self.fig_correlation.clear()
            
            # 1. Matriz de correlación
            ax1 = self.fig_correlation.add_subplot(221)
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                       center=0, ax=ax1)
            ax1.set_title('Matriz de Correlación General')
            
            # 2. Correlaciones con Atención por onda
            correlations = {}
            p_values = {}
            for wave in wave_columns:
                corr, p_value = stats.pearsonr(df_all['Attention'], df_all[wave])
                correlations[wave] = corr
                p_values[wave] = p_value
            
            ax2 = self.fig_correlation.add_subplot(222)
            correlation_df = pd.DataFrame(list(correlations.items()), 
                                       columns=['Onda', 'Correlación'])
            sns.barplot(data=correlation_df, x='Onda', y='Correlación', ax=ax2)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            ax2.set_title('Correlación entre Ondas y Atención')
            
            # 3. Correlaciones por tarea
            f.write("\nCorrelaciones por tarea:\n")
            f.write("="*50 + "\n")
            task_correlations = {}
            for tarea, df in self.dataframes.items():
                task_correlations[tarea] = {}
                f.write(f"\nTarea: {tarea}\n")
                for wave in wave_columns:
                    corr, p_value = stats.pearsonr(df['Attention'], df[wave])
                    task_correlations[tarea][wave] = corr
                    significance = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    f.write(f"{wave}: {corr:.3f} {significance}\n")
            
            ax3 = self.fig_correlation.add_subplot(212)
            task_corr_df = pd.DataFrame(task_correlations).T
            sns.heatmap(task_corr_df, annot=True, cmap='coolwarm', center=0, ax=ax3)
            ax3.set_title('Correlaciones por Tarea')
            
            
            #self.fig_correlation.tight_layout()
            
            #self.canvas_correlation.draw()
            #self.fig_correlation.savefig(plot_
            self.fig_correlation.tight_layout()
            self.canvas_correlation.draw()
            self.fig_correlation.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    def runFFTAnalysis(self, usuario):
        """Ejecuta el análisis FFT y actualiza la pestaña correspondiente"""
        # Configurar archivo de resultados
        plot_filename = os.path.join('Results', usuario, f"GraficaFrec_{usuario}.png")
        
        # Definir bandas de frecuencia
        FREQ_BANDS = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        def calculate_fft(raw_signal):
            """Calcula la FFT de la señal"""
            n = len(raw_signal)
            fft_result = np.fft.fft(raw_signal)
            freqs = np.fft.fftfreq(n, 1/100)  # Usar 100 Hz para visualización
            
            pos_mask = freqs >= 0
            freqs = freqs[pos_mask]
            power = np.abs(fft_result[pos_mask])**2
            
            return freqs, power
        
        # Limpiar figura anterior
        self.fig_fft.clear()
        
        # Crear subplots
        axes = self.fig_fft.subplots(2, 2)
        axes = axes.ravel()
        
        for idx, (nombre, actividad) in enumerate(self.raw_dataframes.items()):
            try:
                # Obtener señal raw
                raw_signal = actividad['Raw_Value'].values
                
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
                
                # Limitar visualización hasta 60 Hz
                ax.set_xlim(0, 60)
                
            except Exception as e:
                print(f"Error procesando {nombre}: {str(e)}")
        
        self.fig_fft.tight_layout()
        self.canvas_fft.draw()
        self.fig_fft.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    def runWaveletAnalysis(self, usuario):
        """Ejecuta el análisis Wavelet y actualiza la pestaña correspondiente"""
        # Configurar archivos de resultados
        results_file = os.path.join('Results', usuario, f"ResultsWav_{usuario}.txt")
        plot_filename = os.path.join('Results', usuario, f"GraficaWav_{usuario}.png")
        
        FREQ_BANDS = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        def calculate_wavelet(signal_data, fs=100):
            """Calcula la transformada wavelet"""
            widths = np.arange(1, 128)
            frequencies = fs * (1 / widths)
            
            wavelets = [signal.morlet2(M=int(w * 10), s=w, w=5.0) for w in widths]
            cwtmatr = np.array([signal.convolve(signal_data, wav, mode='same') 
                               for wav in wavelets])
            
            return cwtmatr, frequencies
        
        with open(results_file, 'w') as f:
            # Limpiar figura anterior
            self.fig_wavelet.clear()
            
            # Crear subplots
            axes = self.fig_wavelet.subplots(2, 2)
            axes = axes.ravel()
            
            for idx, (nombre, df_raw) in enumerate(self.raw_dataframes.items()):
                try:
                    raw_signal = df_raw['Raw_Value'].values
                    
                    # Tomar muestra de 10 segundos
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
                    times = np.arange(len(raw_signal))/100
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
                    
                    ax.set_yscale('log')
                    ax.set_ylim(1, 50)
                    
                    # Calcular y guardar energía por banda
                    f.write(f"\nAnálisis Wavelet para {nombre}:\n")
                    for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
                        mask = (frequencies >= low_freq) & (frequencies <= high_freq)
                        band_energy = np.mean(np.abs(coefficients[mask, :]))
                        f.write(f"Energía en banda {band_name}: {band_energy:.2f}\n")
                    
                except Exception as e:
                    print(f"Error procesando {nombre}: {str(e)}")
            
            self.fig_wavelet.tight_layout()
            self.canvas_wavelet.draw()
            self.fig_wavelet.savefig(plot_filename, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EEGAnalyzer()
    window.show()
    sys.exit(app.exec_())