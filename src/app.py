import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import joblib
import logging
from logging.config import dictConfig

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Importar configuración
from config import (
    GUI_CONFIG, MODEL_FILES, SENTIMENT_LABELS, 
    get_model_path, model_exists, LOGGING_CONFIG
)
from preprocessing import preprocess_pipeline

# Configurar logging
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Paleta de colores moderna
COLORS = {
    'primary': '#2c3e50',       # Azul oscuro
    'secondary': '#3498db',     # Azul brillante
    'success': '#27ae60',       # Verde
    'warning': '#f39c12',       # Naranja
    'danger': '#e74c3c',        # Rojo
    'light': '#ecf0f1',         # Gris claro
    'dark': '#34495e',          # Gris oscuro
    'white': '#ffffff',
    'bg_main': '#f5f6fa',       # Fondo principal
    'bg_section': '#ffffff',    # Fondo de secciones
}


def train_action(root, status_label, train_btn):
    """
    Entrena los modelos de clasificación desde la GUI.
    
    Utiliza el pipeline moderno de train_models.py para entrenar
    Naive Bayes, Logistic Regression y Random Forest.
    """
    path = filedialog.askopenfilename(
        title="Seleccionar archivo CSV con reseñas IMDB", 
        filetypes=[("CSV files", "*.csv"), ("All files", "*")]
    )
    if not path:
        return

    def _train():
        try:
            train_btn.config(state='disabled', text="Entrenando...")
            status_label.config(
                text="⏳ Entrenando modelos... Esto puede tardar varios minutos", 
                fg=COLORS['warning']
            )
            root.update()
            
            logger.info(f"Iniciando entrenamiento desde: {path}")
            
            # Importar módulo de entrenamiento
            from train_models import train_all_models_from_file
            
            # Entrenar todos los modelos
            results = train_all_models_from_file(path)
            
            logger.info("Entrenamiento completado exitosamente")
            
            status_label.config(
                text=f"✓ Modelos entrenados y guardados correctamente", 
                fg=COLORS['success']
            )
            
            # Crear mensaje con resultados
            msg = "✅ Entrenamiento completado exitosamente\n\n"
            msg += "Modelos guardados en la carpeta 'models/':\n\n"
            for model_name in ['naive_bayes', 'logistic_regression', 'random_forest']:
                msg += f"  • {model_name.replace('_', ' ').title()}\n"
            msg += f"  • Vectorizador TF-IDF\n\n"
            msg += "Ya puede clasificar textos usando cualquiera de los modelos."
            
            messagebox.showinfo("Entrenamiento Completo", msg)
            
        except ImportError as e:
            logger.error(f"Error importando módulos: {e}")
            messagebox.showerror(
                "Error de importación", 
                f"No se pudo importar el módulo de entrenamiento:\n{str(e)}\n\n"
                "Verifique que train_models.py existe en la carpeta src/"
            )
            status_label.config(text="✗ Error en entrenamiento", fg=COLORS['danger'])
        except Exception as e:
            logger.exception(f"Error durante entrenamiento: {e}")
            import traceback
            messagebox.showerror(
                "Error al entrenar", 
                f"Ocurrió un error durante el entrenamiento:\n{str(e)}\n\n"
                "Ver consola para más detalles."
            )
            traceback.print_exc()
            status_label.config(text="✗ Error en entrenamiento", fg=COLORS['danger'])
        finally:
            train_btn.config(state='normal', text="Entrenar Modelos")

    threading.Thread(target=_train, daemon=True).start()


def classify_action(text_widget, result_box, classify_btn, model_selector=None):
    """
    Clasifica el texto usando modelos supervisados (Naive Bayes, Logistic Regression, Random Forest).
    
    Esta función reemplaza completamente el sistema legacy basado en similitud coseno.
    Ahora usa modelos de ML supervisados entrenados correctamente.
    """
    text = text_widget.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Texto vacío", "Por favor, escriba o cargue un texto para clasificar")
        return
    
    classify_btn.config(state='disabled', text="Clasificando...")
    logger.info("Iniciando clasificación de texto...")
    
    def _classify():
        try:
            # Determinar qué modelo usar
            model_name_map = {
                'Naive Bayes': 'naive_bayes',
                'Logistic Regression': 'logistic_regression',
                'Random Forest': 'random_forest'
            }
            
            if model_selector and model_selector.get():
                selected_display_name = model_selector.get()
                model_key = model_name_map.get(selected_display_name, 'logistic_regression')
            else:
                model_key = 'logistic_regression'  # Default
            
            logger.info(f"Modelo seleccionado: {model_key}")
            
            # Verificar que el modelo existe
            model_path = get_model_path(model_key)
            vectorizer_path = MODEL_FILES['vectorizer']
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"El modelo '{model_key}' no está entrenado.\n\n"
                    "Por favor, ejecute los notebooks (03_model_training.ipynb) "
                    "o use train_models.py para entrenar los modelos primero."
                )
            
            if not vectorizer_path.exists():
                raise FileNotFoundError(
                    "El vectorizador TF-IDF no existe.\n\n"
                    "Por favor, entrene los modelos usando los notebooks."
                )
            
            # Cargar modelo y vectorizador
            logger.info(f"Cargando modelo desde: {model_path}")
            loaded_model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            
            # Preprocesar texto
            logger.info("Preprocesando texto...")
            text_clean = preprocess_pipeline(text)
            logger.debug(f"Texto preprocesado: {text_clean[:100]}...")
            
            # Vectorizar
            text_vec = vectorizer.transform([text_clean])
            
            # Predecir
            prediction = loaded_model.predict(text_vec)[0]
            
            # Obtener probabilidades
            if hasattr(loaded_model, 'predict_proba'):
                proba = loaded_model.predict_proba(text_vec)[0]
                confidence = max(proba)
                proba_neg = proba[0]
                proba_pos = proba[1]
            else:
                # Para modelos sin predict_proba (ej: algunos SVM)
                confidence = 0.5
                proba_neg = 0.5
                proba_pos = 0.5
            
            sentiment = SENTIMENT_LABELS[prediction]
            logger.info(f"Predicción: {sentiment.upper()} (confianza: {confidence:.2%})")
            
            # Mostrar resultado en la GUI
            result_box.configure(state="normal")
            result_box.delete("1.0", tk.END)
            
            # Configurar tags de formato
            result_box.tag_configure("header", font=("Segoe UI", 14, "bold"), 
                                    foreground=COLORS['primary'], spacing1=10, spacing3=10)
            result_box.tag_configure("positive", font=("Segoe UI", 12, "bold"), 
                                    foreground=COLORS['success'])
            result_box.tag_configure("negative", font=("Segoe UI", 12, "bold"), 
                                    foreground=COLORS['danger'])
            result_box.tag_configure("metric_label", font=("Segoe UI", 10), 
                                    foreground=COLORS['dark'], spacing1=8)
            result_box.tag_configure("metric_value", font=("Segoe UI", 10, "bold"), 
                                    foreground=COLORS['secondary'])
            result_box.tag_configure("separator", font=("Segoe UI", 8), 
                                    foreground=COLORS['light'], spacing1=5, spacing3=5)
            
            # ==== ENCABEZADO ====
            result_box.insert(tk.END, "\n")
            result_box.insert(tk.END, " " * 15 + "🎬 ANÁLISIS DE SENTIMIENTO 🎬\n", "header")
            result_box.insert(tk.END, "\n")
            result_box.insert(tk.END, "─" * 70 + "\n", "separator")
            
            # ==== RESULTADO PRINCIPAL ====
            result_box.insert(tk.END, "\n")
            sentiment_tag = "positive" if prediction == 1 else "negative"
            sentiment_icon = "😊" if prediction == 1 else "�"
            sentiment_text = f"{sentiment_icon}  SENTIMIENTO: {sentiment.upper()}"
            result_box.insert(tk.END, sentiment_text + "\n", sentiment_tag)
            result_box.insert(tk.END, "\n")
            
            # ==== MÉTRICAS ====
            result_box.insert(tk.END, "📊  MÉTRICAS DE CONFIANZA\n", "metric_label")
            result_box.insert(tk.END, "\n")
            
            # Confianza general
            result_box.insert(tk.END, "    Confianza del Modelo: ", "metric_label")
            result_box.insert(tk.END, f"{confidence:.1%}\n", "metric_value")
            
            # Probabilidades individuales
            result_box.insert(tk.END, "\n    Probabilidad Positiva: ", "metric_label")
            result_box.insert(tk.END, f"{proba_pos:.1%}\n", "positive")
            
            result_box.insert(tk.END, "    Probabilidad Negativa: ", "metric_label")
            result_box.insert(tk.END, f"{proba_neg:.1%}\n", "negative")
            
            # Modelo usado
            result_box.insert(tk.END, "\n    Modelo Utilizado: ", "metric_label")
            result_box.insert(tk.END, f"{model_key.replace('_', ' ').title()}\n", "metric_value")
            
            result_box.insert(tk.END, "\n")
            result_box.insert(tk.END, "─" * 70 + "\n", "separator")
            
            # ==== INTERPRETACIÓN ====
            result_box.insert(tk.END, "\n")
            result_box.insert(tk.END, "💡  INTERPRETACIÓN\n", "metric_label")
            result_box.insert(tk.END, "\n")
            
            if confidence > 0.8:
                interpretation = "    El modelo está muy seguro de su predicción."
            elif confidence > 0.6:
                interpretation = "    El modelo tiene confianza moderada en su predicción."
            else:
                interpretation = "    El modelo tiene baja confianza. El texto puede ser ambiguo."
            
            result_box.insert(tk.END, interpretation + "\n", "metric_label")
            result_box.insert(tk.END, "\n")
            
            result_box.configure(state="disabled")
            
            logger.info("Clasificación completada exitosamente")
            
        except FileNotFoundError as e:
            logger.error(f"Modelo no encontrado: {e}")
            messagebox.showerror(
                "Modelo no encontrado",
                str(e)
            )
        except joblib.JoblibException as e:
            logger.error(f"Error cargando modelo corrupto: {e}")
            messagebox.showerror(
                "Modelo corrupto",
                f"El archivo del modelo está dañado o es incompatible:\n{str(e)}\n\n"
                "Por favor, re-entrene el modelo ejecutando los notebooks."
            )
        except KeyError as e:
            logger.error(f"Modelo incompatible: {e}")
            messagebox.showerror(
                "Modelo incompatible",
                f"El modelo no contiene los componentes esperados:\n{str(e)}\n\n"
                "Use modelos entrenados con train_models.py o los notebooks."
            )
        except Exception as e:
            logger.exception(f"Error inesperado al clasificar: {e}")
            import traceback
            messagebox.showerror(
                "Error inesperado",
                f"Error al clasificar:\n{str(e)}\n\nVer consola para detalles."
            )
            print("=" * 70)
            print("TRACEBACK COMPLETO:")
            traceback.print_exc()
            print("=" * 70)
        finally:
            classify_btn.config(state='normal', text="🔍 Clasificar Texto")
    
    threading.Thread(target=_classify, daemon=True).start()


def load_text_file(text_widget):
    path = filedialog.askopenfilename(
        title="Seleccionar archivo de texto", 
        filetypes=[("Text files", "*.txt"), ("All files", "*")]
    )
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, txt)
        messagebox.showinfo("Archivo cargado", f"Texto cargado exitosamente desde:\n{os.path.basename(path)}")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")


def clear_text(text_widget):
    """Limpia el área de texto"""
    if text_widget.get("1.0", tk.END).strip():
        if messagebox.askyesno("Limpiar texto", "¿Está seguro de que desea limpiar el texto?"):
            text_widget.delete("1.0", tk.END)


def create_styled_button(parent, text, command, bg_color, width=20):
    """Crea un botón con estilo consistente"""
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg_color,
        fg=COLORS['white'],
        font=("Segoe UI", 10, "bold"),
        relief=tk.FLAT,
        bd=0,
        padx=15,
        pady=8,
        cursor="hand2",
        width=width,
        activebackground=bg_color,
        activeforeground=COLORS['white']
    )
    
    # Efectos hover
    def on_enter(e):
        btn['bg'] = adjust_color_brightness(bg_color, 0.8)
    
    def on_leave(e):
        btn['bg'] = bg_color
    
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    
    return btn


def adjust_color_brightness(hex_color, factor):
    """Ajusta el brillo de un color hexadecimal"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    rgb = tuple(max(0, min(255, int(c * factor))) for c in rgb)
    return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'


def build_gui():
    root = tk.Tk()
    root.title("🎬 Clasificador de Reseñas de Cine - Machine Learning")
    root.geometry("1400x800")
    root.configure(bg=COLORS['bg_main'])
    root.resizable(True, True)
    
    # Configurar el estilo ttk
    style = ttk.Style()
    style.theme_use('clam')
    
    # ==================== HEADER ====================
    header_frame = tk.Frame(root, bg=COLORS['primary'], height=90)
    header_frame.pack(fill=tk.X)
    header_frame.pack_propagate(False)
    
    title_label = tk.Label(
        header_frame,
        text="🎬 CLASIFICADOR DE RESEÑAS DE CINE",
        font=("Segoe UI", 18, "bold"),
        bg=COLORS['primary'],
        fg=COLORS['white']
    )
    title_label.pack(pady=(15, 5))
    
    subtitle_label = tk.Label(
        header_frame,
        text="Sistema de Machine Learning para identificar reseñas cinematográficas",
        font=("Segoe UI", 10),
        bg=COLORS['primary'],
        fg=COLORS['light']
    )
    subtitle_label.pack(pady=(0, 15))
    
    # ==================== CONTENEDOR PRINCIPAL DE DOS COLUMNAS ====================
    main_container = tk.Frame(root, bg=COLORS['bg_main'])
    main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # ==================== COLUMNA IZQUIERDA - CONTROLES ====================
    left_column = tk.Frame(main_container, bg=COLORS['bg_main'])
    left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
    
    # --- SECCIÓN DE ENTRENAMIENTO ---
    training_frame = tk.Frame(left_column, bg=COLORS['bg_section'], relief=tk.RAISED, bd=1)
    training_frame.pack(fill=tk.X, pady=(0, 15))
    
    training_title = tk.Label(
        training_frame,
        text="⚙️  ENTRENAMIENTO DEL MODELO",
        font=("Segoe UI", 12, "bold"),
        bg=COLORS['bg_section'],
        fg=COLORS['primary']
    )
    training_title.pack(anchor=tk.W, padx=15, pady=(15, 10))
    
    training_info = tk.Label(
        training_frame,
        text="Cargue un archivo CSV con reseñas para entrenar el modelo",
        font=("Segoe UI", 9),
        bg=COLORS['bg_section'],
        fg=COLORS['dark']
    )
    training_info.pack(anchor=tk.W, padx=15, pady=(0, 10))
    
    training_controls = tk.Frame(training_frame, bg=COLORS['bg_section'])
    training_controls.pack(fill=tk.X, padx=15, pady=(0, 15))
    
    btn_train = create_styled_button(
        training_controls,
        "📂 Cargar CSV y Entrenar",
        lambda: train_action(root, status_label, btn_train),
        COLORS['secondary'],
        width=25
    )
    btn_train.pack(side=tk.LEFT, padx=(0, 10))
    
    status_label = tk.Label(
        training_controls,
        text="○ Modelo: no entrenado",
        font=("Segoe UI", 9),
        bg=COLORS['bg_section'],
        fg=COLORS['dark']
    )
    status_label.pack(side=tk.LEFT, padx=5)
    
    # --- SECCIÓN DE CLASIFICACIÓN ---
    classification_frame = tk.Frame(left_column, bg=COLORS['bg_section'], relief=tk.RAISED, bd=1)
    classification_frame.pack(fill=tk.BOTH, expand=True)
    
    classification_title = tk.Label(
        classification_frame,
        text="📝  ÁREA DE CLASIFICACIÓN",
        font=("Segoe UI", 12, "bold"),
        bg=COLORS['bg_section'],
        fg=COLORS['primary']
    )
    classification_title.pack(anchor=tk.W, padx=15, pady=(15, 10))
    
    classification_info = tk.Label(
        classification_frame,
        text="Escriba o pegue el texto que desea clasificar",
        font=("Segoe UI", 9),
        bg=COLORS['bg_section'],
        fg=COLORS['dark']
    )
    classification_info.pack(anchor=tk.W, padx=15, pady=(0, 10))
    
    # Botones de control de texto
    text_controls = tk.Frame(classification_frame, bg=COLORS['bg_section'])
    text_controls.pack(fill=tk.X, padx=15, pady=(0, 10))
    
    btn_loadtxt = create_styled_button(
        text_controls,
        "📄 Cargar Archivo",
        lambda: load_text_file(text_box),
        COLORS['dark'],
        width=18
    )
    btn_loadtxt.pack(side=tk.LEFT, padx=(0, 5))
    
    btn_clear = create_styled_button(
        text_controls,
        "🗑️ Limpiar",
        lambda: clear_text(text_box),
        COLORS['danger'],
        width=15
    )
    btn_clear.pack(side=tk.LEFT)
    
    # Área de texto con scroll
    text_frame = tk.Frame(classification_frame, bg=COLORS['bg_section'])
    text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))
    
    text_box = scrolledtext.ScrolledText(
        text_frame,
        height=15,
        font=("Consolas", 10),
        wrap=tk.WORD,
        relief=tk.SOLID,
        bd=1,
        bg=COLORS['white'],
        fg=COLORS['dark'],
        insertbackground=COLORS['secondary']
    )
    text_box.pack(fill=tk.BOTH, expand=True)
    
    # Botón de clasificación prominente
    btn_classify = create_styled_button(
        classification_frame,
        "🔍 Clasificar Texto",
        lambda: classify_action(text_box, result_box, btn_classify),
        COLORS['success'],
        width=35
    )
    btn_classify.pack(pady=15)
    
    # ==================== COLUMNA DERECHA - RESULTADOS ====================
    right_column = tk.Frame(main_container, bg=COLORS['bg_main'])
    right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
    
    # --- SECCIÓN DE RESULTADOS ---
    results_frame = tk.Frame(right_column, bg=COLORS['bg_section'], relief=tk.RAISED, bd=1)
    results_frame.pack(fill=tk.BOTH, expand=True)
    
    results_title = tk.Label(
        results_frame,
        text="📊  RESULTADOS DEL ANÁLISIS",
        font=("Segoe UI", 14, "bold"),
        bg=COLORS['bg_section'],
        fg=COLORS['primary']
    )
    results_title.pack(anchor=tk.W, padx=15, pady=(15, 10))
    
    results_info = tk.Label(
        results_frame,
        text="Los resultados de la clasificación aparecerán aquí",
        font=("Segoe UI", 9),
        bg=COLORS['bg_section'],
        fg=COLORS['dark']
    )
    results_info.pack(anchor=tk.W, padx=15, pady=(0, 10))
    
    result_box = scrolledtext.ScrolledText(
        results_frame,
        state="disabled",
        font=("Consolas", 10),
        wrap=tk.WORD,
        relief=tk.SOLID,
        bd=1,
        bg=COLORS['white'],
        fg=COLORS['dark']
    )
    result_box.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
    
    # No verificar ni cargar el modelo automáticamente al inicio
    # El usuario debe cargar el modelo manualmente
    status_label.config(text="○ Modelo: no cargado (entrene o cargue el modelo)", fg=COLORS['warning'])
    
    # Footer
    footer_frame = tk.Frame(root, bg=COLORS['primary'], height=35)
    footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
    footer_frame.pack_propagate(False)
    
    footer_label = tk.Label(
        footer_frame,
        text="© 2025 - Sistema de Clasificación ML | UPTC - Inteligencia Computacional",
        font=("Segoe UI", 8),
        bg=COLORS['primary'],
        fg=COLORS['light']
    )
    footer_label.pack(pady=8)
    
    root.mainloop()


if __name__ == "__main__":
    build_gui()
