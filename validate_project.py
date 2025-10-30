#!/usr/bin/env python3
"""
Script de validación completa del proyecto Machine Learning.

Verifica:
- Estructura de carpetas
- Módulos Python importables
- Notebooks existentes
- Dependencias instaladas
- Dataset presente
- Modelos entrenados
- Resultados generados

Uso:
    python validate_project.py           # Validación completa
    python validate_project.py --fix     # Crear carpetas faltantes
    python validate_project.py --verbose # Modo detallado
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict

# Colores para terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg: str):
    """Imprime mensaje de éxito en verde."""
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_warning(msg: str):
    """Imprime advertencia en amarillo."""
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def print_error(msg: str):
    """Imprime error en rojo."""
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_info(msg: str):
    """Imprime información en azul."""
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_header(title: str):
    """Imprime encabezado de sección."""
    separator = "=" * 80
    print(f"\n{Colors.CYAN}{separator}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title:^80}{Colors.RESET}")
    print(f"{Colors.CYAN}{separator}{Colors.RESET}\n")

class ProjectValidator:
    """Validador completo del proyecto."""
    
    def __init__(self, project_dir: Path, verbose: bool = False, fix: bool = False):
        self.project_dir = project_dir
        self.verbose = verbose
        self.fix = fix
        self.errors = []
        self.warnings = []
        self.successes = []
    
    def validate_all(self) -> bool:
        """Ejecuta todas las validaciones."""
        print_header("🔍 VALIDACIÓN DEL PROYECTO")
        
        print_info(f"Directorio del proyecto: {self.project_dir}")
        print_info(f"Modo verbose: {self.verbose}")
        print_info(f"Modo fix: {self.fix}")
        
        # Ejecutar validaciones
        self.validate_folder_structure()
        self.validate_python_modules()
        self.validate_notebooks()
        self.validate_dataset()
        self.validate_dependencies()
        self.validate_models()
        self.validate_results()
        
        # Resumen final
        self.print_summary()
        
        return len(self.errors) == 0
    
    def validate_folder_structure(self):
        """Valida que existan las carpetas necesarias."""
        print_header("📁 VALIDACIÓN: Estructura de Carpetas")
        
        required_folders = {
            'src': 'Módulos de Python',
            'notebooks': 'Notebooks de Jupyter',
            'models': 'Modelos entrenados',
            'results': 'Resultados y visualizaciones',
            'data': 'Datos preprocesados (opcional)'
        }
        
        for folder, description in required_folders.items():
            folder_path = self.project_dir / folder
            
            if folder_path.exists():
                print_success(f"{folder}/ - {description}")
                self.successes.append(f"Carpeta {folder}/ existe")
            else:
                if self.fix:
                    folder_path.mkdir(parents=True, exist_ok=True)
                    print_warning(f"{folder}/ - Creada automáticamente")
                    self.warnings.append(f"Carpeta {folder}/ fue creada")
                else:
                    print_error(f"{folder}/ - NO ENCONTRADA")
                    self.errors.append(f"Carpeta {folder}/ no existe")
    
    def validate_python_modules(self):
        """Valida que existan y se puedan importar los módulos Python."""
        print_header("🐍 VALIDACIÓN: Módulos de Python")
        
        required_modules = {
            'preprocessing': 'Preprocesamiento de texto',
            'train_models': 'Entrenamiento de modelos',
            'evaluation': 'Evaluación y métricas',
            'visualizations': 'Visualizaciones',
            'model': 'Pipeline legacy (GUI)',
            'app': 'Interfaz gráfica',
            'data_preparation': 'Preparación de datos'
        }
        
        src_path = self.project_dir / 'src'
        
        for module_name, description in required_modules.items():
            module_file = src_path / f"{module_name}.py"
            
            if module_file.exists():
                print_success(f"{module_name}.py - {description}")
                self.successes.append(f"Módulo {module_name}.py existe")
            else:
                print_error(f"{module_name}.py - NO ENCONTRADO")
                self.errors.append(f"Módulo {module_name}.py no existe")
        
        # Validar importaciones
        if src_path.exists():
            print_info("Validando importaciones de módulos...")
            sys.path.insert(0, str(src_path))
            
            for module_name in ['preprocessing', 'train_models', 'evaluation', 'visualizations']:
                try:
                    importlib.import_module(module_name)
                    print_success(f"Importación exitosa: {module_name}")
                    self.successes.append(f"Módulo {module_name} importable")
                except Exception as e:
                    print_error(f"Error al importar {module_name}: {str(e)}")
                    self.errors.append(f"Módulo {module_name} no importable")
    
    def validate_notebooks(self):
        """Valida que existan los notebooks de Jupyter."""
        print_header("📓 VALIDACIÓN: Notebooks de Jupyter")
        
        required_notebooks = {
            '01_data_exploration.ipynb': 'Exploración de datos',
            '02_preprocessing.ipynb': 'Preprocesamiento',
            '03_model_training.ipynb': 'Entrenamiento de modelos',
            '04_evaluation.ipynb': 'Evaluación de modelos',
            '05_complete_workflow.ipynb': 'Workflow completo (PRODUCTO FINAL)'
        }
        
        notebooks_path = self.project_dir / 'notebooks'
        
        for notebook, description in required_notebooks.items():
            notebook_path = notebooks_path / notebook
            
            if notebook_path.exists():
                size_kb = notebook_path.stat().st_size / 1024
                print_success(f"{notebook} ({size_kb:.1f} KB) - {description}")
                self.successes.append(f"Notebook {notebook} existe")
            else:
                print_error(f"{notebook} - NO ENCONTRADO")
                self.errors.append(f"Notebook {notebook} no existe")
    
    def validate_dataset(self):
        """Valida que exista el dataset IMDB."""
        print_header("📊 VALIDACIÓN: Dataset IMDB")
        
        dataset_path = self.project_dir / 'IMDB Dataset.csv'
        
        if dataset_path.exists():
            size_mb = dataset_path.stat().st_size / (1024 * 1024)
            print_success(f"Dataset encontrado: IMDB Dataset.csv ({size_mb:.1f} MB)")
            self.successes.append("Dataset IMDB existe")
            
            # Validar tamaño esperado (aproximadamente 63 MB)
            if 60 < size_mb < 70:
                print_info(f"Tamaño correcto (~63 MB esperado)")
            else:
                print_warning(f"Tamaño inesperado: {size_mb:.1f} MB (se esperaban ~63 MB)")
                self.warnings.append(f"Dataset tiene tamaño inesperado: {size_mb:.1f} MB")
        else:
            print_error("IMDB Dataset.csv NO ENCONTRADO")
            self.errors.append("Dataset IMDB no existe")
            print_info("💡 Descargue el dataset de: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    
    def validate_dependencies(self):
        """Valida que requirements.txt exista y contenga los paquetes necesarios."""
        print_header("📦 VALIDACIÓN: Dependencias (requirements.txt)")
        
        req_file = self.project_dir / 'requirements.txt'
        
        if not req_file.exists():
            print_error("requirements.txt NO ENCONTRADO")
            self.errors.append("requirements.txt no existe")
            return
        
        print_success("requirements.txt existe")
        self.successes.append("requirements.txt existe")
        
        # Leer y validar paquetes requeridos
        required_packages = [
            'pandas', 'numpy', 'scikit-learn', 'joblib', 
            'nltk', 'matplotlib', 'seaborn', 'wordcloud',
            'jupyter', 'notebook'
        ]
        
        with open(req_file, 'r') as f:
            requirements_content = f.read().lower()
        
        print_info("Validando paquetes requeridos...")
        for package in required_packages:
            if package in requirements_content:
                print_success(f"  {package}")
                self.successes.append(f"Paquete {package} en requirements.txt")
            else:
                print_error(f"  {package} - FALTANTE")
                self.errors.append(f"Paquete {package} no está en requirements.txt")
    
    def validate_models(self):
        """Valida que existan los modelos entrenados."""
        print_header("🤖 VALIDACIÓN: Modelos Entrenados")
        
        models_path = self.project_dir / 'models'
        
        required_models = [
            'naive_bayes.joblib',
            'logistic_regression.joblib',
            'random_forest.joblib',
            'vectorizer.joblib'
        ]
        
        found_models = 0
        
        for model_file in required_models:
            model_path = models_path / model_file
            
            if model_path.exists():
                size_kb = model_path.stat().st_size / 1024
                print_success(f"{model_file} ({size_kb:.1f} KB)")
                self.successes.append(f"Modelo {model_file} existe")
                found_models += 1
            else:
                print_warning(f"{model_file} NO ENCONTRADO")
                self.warnings.append(f"Modelo {model_file} no existe")
        
        if found_models == 0:
            print_error("Ningún modelo entrenado encontrado")
            self.errors.append("No hay modelos entrenados")
            print_info("💡 Ejecute notebook 03_model_training.ipynb para entrenar modelos")
        elif found_models < len(required_models):
            print_warning(f"Algunos modelos faltan ({found_models}/{len(required_models)})")
            print_info("💡 Ejecute notebook 03_model_training.ipynb para entrenar modelos faltantes")
        else:
            print_success(f"Todos los modelos están entrenados ({found_models}/{len(required_models)})")
    
    def validate_results(self):
        """Valida que existan resultados generados."""
        print_header("📊 VALIDACIÓN: Resultados Generados")
        
        results_path = self.project_dir / 'results'
        
        if not results_path.exists():
            print_error("Carpeta results/ NO ENCONTRADA")
            self.errors.append("Carpeta results/ no existe")
            return
        
        # Contar archivos por tipo
        png_files = list(results_path.glob('*.png'))
        csv_files = list(results_path.glob('*.csv'))
        
        print_info(f"Archivos encontrados en results/:")
        print_success(f"  Imágenes: {len(png_files)}")
        print_success(f"  CSVs: {len(csv_files)}")
        
        if len(png_files) >= 4:
            print_success("Suficientes visualizaciones generadas (≥4)")
            self.successes.append(f"{len(png_files)} visualizaciones generadas")
        else:
            print_warning(f"Pocas visualizaciones generadas ({len(png_files)}/4)")
            self.warnings.append(f"Solo {len(png_files)} visualizaciones generadas")
            print_info("💡 Ejecute notebook 04_evaluation.ipynb para generar visualizaciones")
        
        # Verificar archivo específico de métricas
        metrics_file = results_path / 'metrics_comparison.csv'
        if metrics_file.exists():
            print_success("metrics_comparison.csv encontrado")
            self.successes.append("Métricas exportadas")
        else:
            print_warning("metrics_comparison.csv NO encontrado")
            self.warnings.append("Métricas no exportadas")
    
    def print_summary(self):
        """Imprime resumen final de la validación."""
        print_header("📋 RESUMEN DE VALIDACIÓN")
        
        total_checks = len(self.successes) + len(self.warnings) + len(self.errors)
        
        print(f"Total de validaciones: {total_checks}")
        print_success(f"Éxitos: {len(self.successes)}")
        print_warning(f"Advertencias: {len(self.warnings)}")
        print_error(f"Errores: {len(self.errors)}")
        
        if len(self.errors) == 0 and len(self.warnings) == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 PROYECTO COMPLETAMENTE VÁLIDO 🎉{Colors.RESET}")
            print("El proyecto está listo para uso.")
        elif len(self.errors) == 0:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  PROYECTO VÁLIDO CON ADVERTENCIAS ⚠️{Colors.RESET}")
            print("El proyecto funciona pero tiene advertencias menores.")
            
            if self.warnings:
                print(f"\nAdvertencias:")
                for i, warning in enumerate(self.warnings[:5], 1):  # Mostrar máximo 5
                    print(f"  {i}. {warning}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ PROYECTO CON ERRORES ❌{Colors.RESET}")
            print("El proyecto tiene errores que deben corregirse.")
            
            if self.errors:
                print(f"\nErrores críticos:")
                for i, error in enumerate(self.errors[:5], 1):  # Mostrar máximo 5
                    print(f"  {i}. {error}")
            
            if self.fix:
                print(f"\n💡 Algunas carpetas faltantes fueron creadas automáticamente.")
                print(f"Ejecute nuevamente sin --fix para validar el progreso.")

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validador completo del proyecto Machine Learning')
    parser.add_argument('--verbose', action='store_true', help='Modo detallado')
    parser.add_argument('--fix', action='store_true', help='Crear carpetas faltantes automáticamente')
    
    args = parser.parse_args()
    
    # Detectar directorio del proyecto
    project_dir = Path(__file__).parent
    
    # Crear validador y ejecutar
    validator = ProjectValidator(project_dir, verbose=args.verbose, fix=args.fix)
    success = validator.validate_all()
    
    # Retornar código de salida apropiado
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
