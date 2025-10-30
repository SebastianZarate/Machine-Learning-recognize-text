"""
Configuración para pytest.

Este archivo configura pytest para el proyecto.
"""

import sys
from pathlib import Path

# Agregar src al PYTHONPATH
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))
