# Sistema simple para reconocer reseñas de cine (prototipo)

Este proyecto contiene una pequeña aplicación en Python con GUI que permite:

- Cargar un archivo CSV (por ejemplo `IMDB Dataset.csv`) y entrenar un vectorizador TF-IDF sobre las reseñas.
- Pegar o escribir un texto y clasificar si corresponde o no a una reseña de cine.

Características principales:

- Método combinado: coincidencia de palabras clave + similitud coseno contra el vector promedio de reseñas (corpus IMDB). Esto permite decidir si un texto "coincide" con el estilo/tema de reseña de cine.
- Interfaz gráfica (Tkinter) para carga, entrenamiento y clasificación.

Cómo ejecutar (Windows PowerShell):

1. Crear e activar un entorno virtual (opcional pero recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

3. Ejecutar la aplicación (desde la raíz del proyecto):

```powershell
python main.py
```

Uso:

- Pulsar "Cargar CSV y entrenar modelo" y escoger `IMDB Dataset.csv` (o cualquier CSV con columna `review`).
- Pegar o escribir el texto en la caja y pulsar "Clasificar". La aplicación mostrará: decisión (reseña / no reseña), probabilidad combinada, similitud y palabras clave coincidentes.

Notas sobre validación de palabras (pregunta para la sustentación):

- Las palabras clave se definieron manualmente (ej. "película", "director", "actor", "trama", "reseña", "escena", "guion", etc.).
- Para validar su efecto se combinan con una medida estadística: la similitud coseno entre el texto y el vector promedio del corpus de reseñas. De ese modo las palabras clave aportan una señal interpretable (qué palabras aparecen), y la similitud aporta una señal global de estilo/tema.
- En la aplicación se muestran las palabras coincidentes y la contribución (porcentaje) que suman al resultado final; en la presentación se puede mostrar cómo cambian las decisiones al añadir/quitar palabras clave o al variar umbrales.

Posibles mejoras (siguientes pasos):

- Entrenar un clasificador binario real (necesita corpus negativo: textos que no sean reseñas).
- Añadir explicabilidad (LIME/SHAP) para ver qué frases concretas influyen.
- Guardar y versionar modelos, y añadir tests automáticos.