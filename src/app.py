import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# Asegurar que src esté en sys.path cuando se ejecute desde la raíz
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import model


MODEL_PATH = os.path.join(os.path.dirname(script_dir), "models", "review_model.joblib")


def train_action(root, status_label):
    path = filedialog.askopenfilename(title="Seleccionar CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*")])
    if not path:
        return

    def _train():
        try:
            status_label.config(text="Entrenando... esto puede tardar unos segundos")
            model_path = os.path.join(os.path.dirname(script_dir), "models", "review_model.joblib")
            model.train_from_csv(path, model_path=model_path)
            status_label.config(text=f"Modelo entrenado y guardado en: {model_path}")
            messagebox.showinfo("Listo", f"Modelo entrenado y guardado en:\n{model_path}")
        except Exception as e:
            messagebox.showerror("Error al entrenar", str(e))
            status_label.config(text="Error en entrenamiento")

    threading.Thread(target=_train, daemon=True).start()


def classify_action(text_widget, result_box):
    text = text_widget.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Texto vacío", "Escribe o pega un texto para clasificar")
        return
    model_path = os.path.join(os.path.dirname(script_dir), "models", "review_model.joblib")
    try:
        res = model.predict_text(text, model_path=model_path)
    except FileNotFoundError:
        messagebox.showwarning("Modelo no encontrado", "Entrene primero el modelo con un CSV (botón 'Cargar CSV y entrenar modelo').")
        return
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return

    result_box.configure(state="normal")
    result_box.delete("1.0", tk.END)
    decision = "ES una reseña de cine" if res["is_review"] else "NO es una reseña de cine"
    result_box.insert(tk.END, f"Decisión: {decision}\n")
    result_box.insert(tk.END, f"Probabilidad combinada: {res['probability']:.3f}\n")
    result_box.insert(tk.END, f"Similitud con corpus de reseñas: {res['similarity']:.3f}\n")
    result_box.insert(tk.END, f"Puntaje por palabras clave: {res['keyword_score']:.3f}\n")
    if res["matched_keywords"]:
        result_box.insert(tk.END, "Palabras clave encontradas: " + ", ".join(res["matched_keywords"]) + "\n")
    else:
        result_box.insert(tk.END, "Palabras clave encontradas: (ninguna)\n")
    result_box.configure(state="disabled")


def load_text_file(text_widget):
    path = filedialog.askopenfilename(title="Seleccionar archivo de texto", filetypes=[("Text files", "*.txt"), ("All files", "*")])
    if not path:
        return
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    text_widget.delete("1.0", tk.END)
    text_widget.insert(tk.END, txt)


def build_gui():
    root = tk.Tk()
    root.title("Clasificador simple: reseña de cine")
    root.geometry("800x600")

    top_frame = tk.Frame(root)
    top_frame.pack(fill=tk.X, padx=8, pady=8)

    btn_train = tk.Button(top_frame, text="Cargar CSV y entrenar modelo", command=lambda: train_action(root, status_label))
    btn_train.pack(side=tk.LEFT)

    btn_loadtxt = tk.Button(top_frame, text="Cargar archivo de texto", command=lambda: load_text_file(text_box))
    btn_loadtxt.pack(side=tk.LEFT, padx=8)

    status_label = tk.Label(top_frame, text="Modelo: pendiente")
    status_label.pack(side=tk.RIGHT)

    mid_frame = tk.Frame(root)
    mid_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    lbl = tk.Label(mid_frame, text="Escribe o pega aquí el texto a clasificar:")
    lbl.pack(anchor=tk.W)

    global text_box
    text_box = scrolledtext.ScrolledText(mid_frame, height=12)
    text_box.pack(fill=tk.BOTH, expand=True)

    btn_classify = tk.Button(root, text="Clasificar", command=lambda: classify_action(text_box, result_box))
    btn_classify.pack(pady=6)

    lbl_res = tk.Label(root, text="Resultado:")
    lbl_res.pack(anchor=tk.W, padx=8)

    global result_box
    result_box = scrolledtext.ScrolledText(root, height=8, state="disabled")
    result_box.pack(fill=tk.BOTH, expand=False, padx=8, pady=4)

    # mostrar modelo si existe
    model_path = os.path.join(os.path.dirname(script_dir), "models", "review_model.joblib")
    if os.path.exists(model_path):
        status_label.config(text=f"Modelo cargado: {model_path}")
    else:
        status_label.config(text="Modelo: no entrenado")

    root.mainloop()


if __name__ == "__main__":
    build_gui()
