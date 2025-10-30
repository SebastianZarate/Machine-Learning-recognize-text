import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import joblib

# Asegurar que src esté en sys.path cuando se ejecute desde la raíz
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import model


MODEL_PATH = os.path.join(os.path.dirname(script_dir), "models", "review_model.joblib")

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
    path = filedialog.askopenfilename(
        title="Seleccionar archivo CSV con reseñas", 
        filetypes=[("CSV files", "*.csv"), ("All files", "*")]
    )
    if not path:
        return

    def _train():
        try:
            train_btn.config(state='disabled', text="Entrenando...")
            status_label.config(
                text="⏳ Entrenando modelo... Esto puede tardar unos segundos", 
                fg=COLORS['warning']
            )
            root.update()
            
            model_path = os.path.join(os.path.dirname(script_dir), "models", "review_model.joblib")
            model.train_from_csv(path, model_path=model_path)
            
            status_label.config(
                text=f"✓ Modelo entrenado y listo para usar", 
                fg=COLORS['success']
            )
            messagebox.showinfo(
                "Entrenamiento Completo", 
                f"El modelo ha sido entrenado y guardado exitosamente en:\n{model_path}\n\nYa puede clasificar textos."
            )
        except Exception as e:
            messagebox.showerror("Error al entrenar", f"Ocurrió un error durante el entrenamiento:\n{str(e)}")
            status_label.config(text="✗ Error en entrenamiento", fg=COLORS['danger'])
        finally:
            train_btn.config(state='normal', text="Entrenar Modelo")

    threading.Thread(target=_train, daemon=True).start()


def classify_action(text_widget, result_box, classify_btn, model_selector=None):
    text = text_widget.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Texto vacío", "Por favor, escriba o cargue un texto para clasificar")
        return
    
    classify_btn.config(state='disabled', text="Clasificando...")
    
    def _classify():
        try:
            # Determinar qué modelo usar
            if model_selector and model_selector.get():
                selected_model = model_selector.get()
                model_files = {
                    'Naive Bayes': 'naive_bayes.joblib',
                    'Logistic Regression': 'logistic_regression.joblib',
                    'Random Forest': 'random_forest.joblib'
                }
                model_name = model_files.get(selected_model, 'logistic_regression.joblib')
                model_path = os.path.join(os.path.dirname(script_dir), "models", model_name)
                
                # Intentar usar modelo nuevo
                if os.path.exists(model_path):
                    vectorizer_path = os.path.join(os.path.dirname(script_dir), "models", "vectorizer.joblib")
                    if os.path.exists(vectorizer_path):
                        # Cargar modelo y vectorizador modernos
                        loaded_model = joblib.load(model_path)
                        vectorizer = joblib.load(vectorizer_path)
                        
                        # Preprocesar y clasificar
                        from preprocessing import preprocess_pipeline
                        text_clean = preprocess_pipeline(text)
                        text_vec = vectorizer.transform([text_clean])
                        prediction = loaded_model.predict(text_vec)[0]
                        
                        # Obtener probabilidades
                        if hasattr(loaded_model, 'predict_proba'):
                            proba = loaded_model.predict_proba(text_vec)[0]
                        else:
                            proba = [0.5, 0.5]  # Fallback
                        
                        res = {
                            'label': 'Positive' if prediction == 1 else 'Negative',
                            'confidence': max(proba),
                            'proba_neg': proba[0],
                            'proba_pos': proba[1],
                            'model_used': selected_model
                        }
                    else:
                        # Fallback a modelo legacy
                        model_path = os.path.join(os.path.dirname(script_dir), "models", "review_model.joblib")
                        res = model.predict_text(text, model_path=model_path)
                        res['model_used'] = 'Legacy (Cosine Similarity)'
                else:
                    # Fallback a modelo legacy
                    model_path = os.path.join(os.path.dirname(script_dir), "models", "review_model.joblib")
                    res = model.predict_text(text, model_path=model_path)
                    res['model_used'] = 'Legacy (Cosine Similarity)'
            else:
                # Sin selector, usar modelo legacy
                model_path = os.path.join(os.path.dirname(script_dir), "models", "review_model.joblib")
                res = model.predict_text(text, model_path=model_path)
                res['model_used'] = 'Legacy'
            
            result_box.configure(state="normal")
            result_box.delete("1.0", tk.END)
            
            # Configurar tags con mejor formato y espaciado
            result_box.tag_configure("header", font=("Segoe UI", 13, "bold"), foreground=COLORS['primary'], spacing1=10, spacing3=10)
            result_box.tag_configure("section_title", font=("Segoe UI", 11, "bold"), foreground=COLORS['dark'], spacing1=15, spacing3=8)
            result_box.tag_configure("positive", font=("Segoe UI", 11, "bold"), foreground=COLORS['success'])
            result_box.tag_configure("negative", font=("Segoe UI", 11, "bold"), foreground=COLORS['danger'])
            result_box.tag_configure("metric_label", font=("Segoe UI", 10), foreground=COLORS['dark'], spacing1=8)
            result_box.tag_configure("metric_value", font=("Segoe UI", 10, "bold"), foreground=COLORS['secondary'])
            result_box.tag_configure("keywords", font=("Segoe UI", 9), foreground=COLORS['secondary'], spacing1=5, lmargin1=30)
            result_box.tag_configure("separator", font=("Segoe UI", 8), foreground=COLORS['light'], spacing1=5, spacing3=5)
            result_box.tag_configure("box", background="#f8f9fa", relief="solid", borderwidth=1, spacing1=10, spacing3=10, lmargin1=15, lmargin2=15, rmargin=15)
            
            # ==== ENCABEZADO ====
            result_box.insert(tk.END, "\n")
            result_box.insert(tk.END, " " * 18 + "RESULTADO DE CLASIFICACIÓN\n", "header")
            result_box.insert(tk.END, "\n")
            
            # ==== RESULTADO PRINCIPAL ====
            decision = "ES UNA RESEÑA DE CINE ✓" if res["is_review"] else "NO ES UNA RESEÑA DE CINE ✗"
            result_tag = "positive" if res["is_review"] else "negative"
            icon = "🎬" if res["is_review"] else "📄"
            
            result_box.insert(tk.END, f"{icon}  {decision}\n", result_tag)
            result_box.insert(tk.END, "\n")
            result_box.insert(tk.END, "─" * 70 + "\n", "separator")
            
            # ==== MÉTRICAS DE ANÁLISIS ====
            result_box.insert(tk.END, "\n")
            result_box.insert(tk.END, "  MÉTRICAS DE ANÁLISIS\n", "section_title")
            result_box.insert(tk.END, "\n")
            
            # Probabilidad combinada
            prob_color = "positive" if res['probability'] > 0.5 else "negative"
            result_box.insert(tk.END, "    🎯  Probabilidad Combinada:\n", "metric_label")
            result_box.insert(tk.END, f"         {res['probability']:.1%}\n", prob_color)
            
            # Similitud con corpus
            result_box.insert(tk.END, "\n    📊  Similitud con Corpus de Reseñas:\n", "metric_label")
            result_box.insert(tk.END, f"         {res['similarity']:.1%}\n", "metric_value")
            
            # Puntaje por palabras clave
            result_box.insert(tk.END, "\n    🔑  Puntaje por Palabras Clave:\n", "metric_label")
            result_box.insert(tk.END, f"         {res['keyword_score']:.1%}\n", "metric_value")
            
            result_box.insert(tk.END, "\n")
            result_box.insert(tk.END, "─" * 70 + "\n", "separator")
            
            # ==== PALABRAS CLAVE DETECTADAS ====
            result_box.insert(tk.END, "\n")
            result_box.insert(tk.END, "  PALABRAS CLAVE DETECTADAS\n", "section_title")
            result_box.insert(tk.END, "\n")
            
            if res["matched_keywords"]:
                result_box.insert(tk.END, "    🔍  Términos encontrados:\n\n", "metric_label")
                # Mostrar palabras clave de forma organizada
                keywords_formatted = "         • " + "\n         • ".join(res["matched_keywords"])
                result_box.insert(tk.END, keywords_formatted + "\n", "keywords")
            else:
                result_box.insert(tk.END, "    ⚠️  No se detectaron palabras clave específicas de reseñas\n", "metric_label")
            
            result_box.insert(tk.END, "\n")
            result_box.insert(tk.END, "─" * 70 + "\n", "separator")
            result_box.insert(tk.END, "\n")
            
            result_box.configure(state="disabled")
            
        except FileNotFoundError:
            messagebox.showwarning(
                "Modelo no encontrado", 
                "Debe entrenar el modelo primero.\n\nUse el botón 'Entrenar Modelo' y seleccione un archivo CSV con reseñas."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Error al clasificar el texto:\n{str(e)}")
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
