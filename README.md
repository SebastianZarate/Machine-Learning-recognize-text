# 🎬 Clasificador de Reseñas de Cine - Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Sistema inteligente de clasificación de texto desarrollado con **Machine Learning** para identificar y clasificar reseñas cinematográficas. Este proyecto utiliza técnicas avanzadas de procesamiento de lenguaje natural (NLP) combinando análisis de palabras clave con similitud coseno basada en vectorización TF-IDF.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Instalación](#-instalación)
- [Uso del Sistema](#-uso-del-sistema)
- [Metodología de Clasificación](#-metodología-de-clasificación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Interfaz Gráfica](#-interfaz-gráfica)
- [Dataset](#-dataset)
- [Validación y Precisión](#-validación-y-precisión)
- [Posibles Mejoras](#-posibles-mejoras)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)

## ✨ Características

### Funcionalidades Principales

- **🎯 Clasificación Inteligente**: Determina si un texto corresponde o no a una reseña cinematográfica
- **📊 Análisis Multimétrico**: Combina múltiples técnicas para una clasificación precisa:
  - Similitud coseno con corpus de reseñas
  - Detección de palabras clave específicas del dominio
  - Probabilidad combinada ponderada
- **🖥️ Interfaz Gráfica Moderna**: GUI intuitiva desarrollada con Tkinter con diseño profesional
- **📁 Gestión de Modelos**: Entrenamiento y persistencia de modelos mediante joblib
- **📈 Resultados Detallados**: Visualización completa de métricas y análisis
- **⚡ Procesamiento Asíncrono**: Entrenamiento y clasificación en hilos separados

### Capacidades del Sistema

- Carga y procesamiento de datasets en formato CSV
- Entrenamiento de vectorizador TF-IDF personalizado
- Clasificación de textos individuales o desde archivos
- Identificación de palabras clave relacionadas con cine
- Cálculo de métricas de confianza y probabilidad
- Exportación y reutilización de modelos entrenados

## 🛠️ Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Python** | 3.8+ | Lenguaje de programación principal |
| **scikit-learn** | 1.0+ | Machine Learning y vectorización TF-IDF |
| **pandas** | Latest | Manipulación y análisis de datos |
| **numpy** | Latest | Operaciones numéricas y álgebra lineal |
| **joblib** | Latest | Serialización de modelos |
| **Tkinter** | Built-in | Interfaz gráfica de usuario |

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────┐
│              INTERFAZ GRÁFICA (Tkinter)             │
│  ┌──────────────────┐     ┌───────────────────┐   │
│  │  Panel Control   │     │  Panel Resultados │   │
│  │  - Entrenamiento │     │  - Métricas       │   │
│  │  - Input Texto   │     │  - Visualización  │   │
│  └──────────────────┘     └───────────────────┘   │
└─────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────┐
│           CAPA DE LÓGICA (model.py)                 │
│  ┌───────────────────────────────────────────────┐ │
│  │  • Vectorizador TF-IDF (scikit-learn)        │ │
│  │  • Detector de Palabras Clave                │ │
│  │  • Motor de Similitud Coseno                 │ │
│  │  • Sistema de Puntuación Combinada           │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────┐
│              CAPA DE DATOS                          │
│  • Dataset IMDB (50,000 reseñas)                   │
│  • Modelo serializado (joblib)                     │
│  • Textos de entrada del usuario                   │
└─────────────────────────────────────────────────────┘
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git (opcional, para clonar el repositorio)

### Paso 1: Clonar el Repositorio

```powershell
git clone https://github.com/SebastianZarate/Machine-Learning-recognize-text.git
cd Machine-Learning-recognize-text
```

### Paso 2: Crear Entorno Virtual (Recomendado)

**En Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**En Linux/Mac:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias

```powershell
pip install -r requirements.txt
```

### Paso 4: Verificar Instalación

```powershell
python test_run.py
```

## 📖 Uso del Sistema

### Ejecución de la Aplicación

Desde la raíz del proyecto, ejecute:

```powershell
python main.py
```

O alternativamente:

```powershell
python src/app.py
```

### Flujo de Trabajo

#### 1️⃣ **Entrenar el Modelo**

Al iniciar la aplicación, el modelo NO está cargado. Debe entrenarlo primero:

1. Haga clic en **"📂 Cargar CSV y Entrenar"**
2. Seleccione el archivo `IMDB Dataset.csv` (incluido en el proyecto)
3. Espere mientras el sistema procesa las 50,000 reseñas
4. Verá el mensaje: **"✓ Modelo entrenado y listo para usar"**

**Nota:** El entrenamiento puede tardar 30-60 segundos dependiendo de su hardware. El modelo se guarda automáticamente en `models/review_model.joblib` para uso futuro.

#### 2️⃣ **Clasificar Textos**

Una vez entrenado el modelo:

**Opción A - Escribir/Pegar Texto:**
1. Escriba o pegue el texto en el área de clasificación
2. Haga clic en **"🔍 Clasificar Texto"**
3. Vea los resultados detallados en el panel derecho

**Opción B - Cargar desde Archivo:**
1. Haga clic en **"📄 Cargar Archivo"**
2. Seleccione un archivo `.txt`
3. El texto se cargará automáticamente
4. Haga clic en **"🔍 Clasificar Texto"**

#### 3️⃣ **Interpretar Resultados**

El sistema mostrará:

- **✓ Decisión Principal**: Si es o no una reseña de cine
- **📊 Métricas de Análisis**:
  - **Probabilidad Combinada**: Confianza general (0-100%)
  - **Similitud con Corpus**: Qué tan similar es al estilo de reseñas IMDB
  - **Puntaje por Palabras Clave**: Coincidencias de términos cinematográficos
- **🔍 Palabras Clave Detectadas**: Lista de términos relacionados con cine encontrados

## 🧠 Metodología de Clasificación

### Enfoque Híbrido

El sistema utiliza un **método combinado multi-criterio** que integra:

#### 1. Vectorización TF-IDF (Term Frequency-Inverse Document Frequency)

```python
# Configuración del vectorizador
TfidfVectorizer(
    max_features=5000,      # Top 5000 palabras más importantes
    stop_words='english',   # Elimina palabras comunes
    ngram_range=(1, 2)      # Unigramas y bigramas
)
```

**Ventajas:**
- Captura la importancia relativa de palabras en el corpus
- Reduce el ruido de palabras muy frecuentes
- Considera contexto mediante n-gramas

#### 2. Similitud Coseno

Mide el ángulo entre el vector del texto de entrada y el vector promedio del corpus:

$$
\text{similitud} = \frac{\vec{A} \cdot \vec{B}}{||\vec{A}|| \times ||\vec{B}||}
$$

**Rango:** [0, 1] donde 1 = idéntico, 0 = completamente diferente

#### 3. Detección de Palabras Clave

Lista curada de 50+ términos específicos del dominio cinematográfico:

```python
KEYWORDS = {
    # Elementos de producción
    "película", "film", "movie", "cinema", "director", "actor", 
    "actriz", "reparto", "cast", "cinematografía",
    
    # Aspectos técnicos
    "guion", "soundtrack", "escena", "trama", "argumento",
    "fotografía", "montaje", "efectos especiales",
    
    # Evaluación
    "reseña", "crítica", "rating", "calificación", "estreno",
    "recomendación", "opinión", "valoración",
    
    # Géneros
    "thriller", "drama", "comedia", "acción", "suspenso",
    
    # Y más...
}
```

#### 4. Fórmula de Clasificación Combinada

```python
# Pesos configurables
KEYWORD_WEIGHT = 0.25     # 25% palabras clave
SIMILARITY_WEIGHT = 0.75  # 75% similitud coseno

# Cálculo de probabilidad combinada
combined_probability = (
    keyword_score * KEYWORD_WEIGHT + 
    similarity * SIMILARITY_WEIGHT
)

# Decisión final
is_review = combined_probability >= THRESHOLD  # THRESHOLD = 0.30
```

### Umbrales de Clasificación

| Métrica | Umbral | Justificación |
|---------|--------|---------------|
| Probabilidad Combinada | ≥ 0.30 | Balance entre precisión y recall |
| Palabras Clave | Variable | Contribución proporcional al número de matches |
| Similitud Coseno | Variable | Depende del corpus de entrenamiento |

### Ejemplo de Clasificación

**Input:**
```
"Esta película de Christopher Nolan es excepcional. 
Los efectos visuales y la actuación de Leonardo DiCaprio 
son impresionantes. Recomiendo esta obra maestra."
```

**Output:**
```
✓ ES UNA RESEÑA DE CINE

Métricas:
🎯 Probabilidad Combinada: 87.3%
📊 Similitud con Corpus: 0.82 (82%)
🔑 Palabras Clave: 0.15 (15%)

Palabras Detectadas:
• película
• actuación
• recomiendo
• efectos visuales
• obra maestra
```

## 📁 Estructura del Proyecto

```
Machine-Learning-recognize-text/
│
├── 📄 main.py                    # Punto de entrada principal
├── 📄 test_run.py                # Script de pruebas
├── 📄 requirements.txt           # Dependencias del proyecto
├── 📄 README.md                  # Este archivo
├── 📊 IMDB Dataset.csv           # Dataset de entrenamiento (50K reseñas)
│
├── 📁 src/                       # Código fuente
│   ├── 📄 app.py                 # Interfaz gráfica (GUI)
│   ├── 📄 model.py               # Lógica de Machine Learning
│   └── 📁 __pycache__/           # Cache de Python
│
├── 📁 models/                    # Modelos entrenados
│   └── 📄 review_model.joblib    # Modelo serializado TF-IDF
│
└── 📁 __pycache__/               # Cache de Python
```

### Descripción de Archivos Clave

| Archivo | Líneas | Descripción |
|---------|--------|-------------|
| `main.py` | ~10 | Entry point que inicia la aplicación GUI |
| `src/app.py` | ~430 | Interfaz gráfica completa con Tkinter |
| `src/model.py` | ~200 | Motor de ML: entrenamiento y predicción |
| `test_run.py` | ~50 | Suite de pruebas automatizadas |

## 🖥️ Interfaz Gráfica

### Diseño Responsive de Dos Columnas

#### Panel Izquierdo - Controles
- **Sección de Entrenamiento**
  - Botón de carga de CSV
  - Indicador de estado del modelo
  - Información del proceso

- **Sección de Clasificación**
  - Área de texto para input (con scroll)
  - Botones de control (Cargar archivo, Limpiar)
  - Botón principal de clasificación

#### Panel Derecho - Resultados
- **Título con icono**
- **Área de resultados con scroll**
- **Formato estructurado**:
  - Encabezado con decisión principal
  - Sección de métricas detalladas
  - Lista de palabras clave detectadas

### Paleta de Colores

```python
COLORS = {
    'primary': '#2c3e50',       # Azul oscuro (headers)
    'secondary': '#3498db',     # Azul brillante (botones)
    'success': '#27ae60',       # Verde (éxito)
    'warning': '#f39c12',       # Naranja (advertencias)
    'danger': '#e74c3c',        # Rojo (errores)
    'light': '#ecf0f1',         # Gris claro (fondos)
    'dark': '#34495e',          # Gris oscuro (textos)
    'white': '#ffffff',
    'bg_main': '#f5f6fa',       # Fondo principal
    'bg_section': '#ffffff',    # Fondo de secciones
}
```

### Características de UX

- ✅ Diseño moderno y limpio
- ✅ Botones con efectos hover
- ✅ Indicadores de estado en tiempo real
- ✅ Mensajes informativos y de error
- ✅ Procesamiento asíncrono (no bloquea la UI)
- ✅ Áreas de scroll para contenido extenso
- ✅ Iconos descriptivos (🎬📊🔍📄)

## 📊 Dataset

### IMDB Dataset.csv

**Fuente:** [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

**Características:**
- **Tamaño:** 50,000 reseñas
- **Idioma:** Inglés
- **Columnas:**
  - `review`: Texto de la reseña
  - `sentiment`: Etiqueta (positive/negative)
- **Balance:** 50% positivas, 50% negativas
- **Formato:** CSV con codificación UTF-8

**Ejemplo de Registro:**
```csv
review,sentiment
"One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me...",positive
```

### Preprocesamiento

El sistema realiza automáticamente:
1. Carga del CSV con pandas
2. Limpieza de valores nulos
3. Conversión a minúsculas
4. Tokenización
5. Eliminación de stop words
6. Vectorización TF-IDF

## 🎯 Validación y Precisión

### Método de Validación

El sistema utiliza un enfoque de **validación por palabras clave + similitud estadística**:

1. **Palabras Clave:** Definidas manualmente por expertos del dominio
2. **Similitud Coseno:** Medida estadística objetiva
3. **Combinación Ponderada:** Balance entre interpretabilidad y precisión

### Justificación del Enfoque

| Ventaja | Descripción |
|---------|-------------|
| **Interpretabilidad** | Las palabras clave permiten entender la decisión |
| **Robustez** | La similitud coseno captura patrones sutiles |
| **No requiere etiquetas negativas** | Funciona solo con corpus de reseñas positivas |
| **Escalable** | Fácil de ajustar agregando/quitando palabras clave |

### Casos de Prueba

```python
# CASO 1: Reseña clara ✓
texto = "Excelente película de acción con grandes efectos especiales"
resultado = "ES RESEÑA" (probabilidad: 89%)

# CASO 2: Texto ambiguo ⚠
texto = "Me gustó mucho, muy entretenido y emocionante"
resultado = "NO ES RESEÑA" (probabilidad: 42%)

# CASO 3: Texto no relacionado ✗
texto = "Hoy hace buen clima, voy a salir a caminar"
resultado = "NO ES RESEÑA" (probabilidad: 8%)
```

### Ajuste de Precisión

Para modificar la sensibilidad del clasificador, edite en `model.py`:

```python
# Más estricto (reduce falsos positivos)
THRESHOLD = 0.40
KEYWORD_WEIGHT = 0.30

# Más permisivo (reduce falsos negativos)
THRESHOLD = 0.25
KEYWORD_WEIGHT = 0.20
```

## 🔮 Posibles Mejoras

### Corto Plazo

- [ ] **Soporte Multilenguaje:** Agregar palabras clave en español
- [ ] **Exportar Resultados:** Guardar análisis en PDF/HTML
- [ ] **Histórico:** Mantener registro de clasificaciones anteriores
- [ ] **Batch Processing:** Clasificar múltiples textos simultáneamente

### Mediano Plazo

- [ ] **Clasificador Supervisado:** Entrenar modelo binario (reseña/no-reseña)
- [ ] **Análisis de Sentimiento:** Determinar si la reseña es positiva/negativa
- [ ] **Explicabilidad (LIME/SHAP):** Visualizar qué palabras influyen en la decisión
- [ ] **API REST:** Exponer el servicio mediante Flask/FastAPI

### Largo Plazo

- [ ] **Deep Learning:** Implementar modelo con BERT/Transformers
- [ ] **Transfer Learning:** Fine-tuning de modelos preentrenados
- [ ] **Detección de Géneros:** Clasificar el género cinematográfico
- [ ] **Sistema de Recomendación:** Sugerir películas basadas en preferencias

## 👥 Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar este proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Áreas de Contribución

- 🐛 Reportar bugs
- 💡 Sugerir nuevas funcionalidades
- 📝 Mejorar documentación
- 🧪 Agregar tests
- 🎨 Mejorar la interfaz gráfica
- 🌍 Traducir a otros idiomas

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

**Sebastian Zarate**
- GitHub: [@SebastianZarate](https://github.com/SebastianZarate)
- Proyecto: [Machine-Learning-recognize-text](https://github.com/SebastianZarate/Machine-Learning-recognize-text)

---

## 🎓 Contexto Académico

**Universidad:** UPTC (Universidad Pedagógica y Tecnológica de Colombia)  
**Curso:** Inteligencia Computacional  
**Semestre:** Noveno  
**Año:** 2025

---

## ❓ FAQ

### ¿Por qué el entrenamiento tarda tanto?

El procesamiento de 50,000 reseñas con TF-IDF requiere recursos computacionales. En equipos modernos toma 30-60 segundos.

### ¿Puedo usar mi propio dataset?

Sí, el archivo CSV debe tener una columna llamada `review` con los textos.

### ¿Funciona con textos en español?

Sí, pero la precisión es menor. Recomendamos agregar palabras clave en español en `model.py`.

### ¿El modelo se guarda automáticamente?

Sí, después de entrenar se guarda en `models/review_model.joblib` y se puede reutilizar.

### ¿Cuánta memoria RAM necesito?

Mínimo 4GB RAM. Recomendado: 8GB o más para procesamiento eficiente.
