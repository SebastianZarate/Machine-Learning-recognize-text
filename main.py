"""Arrancador simple para ejecutar la aplicación con `python main.py`.

Coloca `src/` en sys.path e invoca `build_gui()` desde `src/app.py`.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import app
except Exception as e:
    print(f"Error al importar el módulo 'app' desde {SRC_DIR}: {e}")
    raise


def main():
    # Llamamos a la construcción de la GUI; `app.build_gui()` bloquea en su bucle principal.
    app.build_gui()


if __name__ == "__main__":
    main()
