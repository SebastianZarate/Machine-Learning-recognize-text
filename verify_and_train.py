"""
Script simple para verificar y entrenar modelos.
"""
import sys
import os
sys.path.insert(0, 'src')

print("="*70)
print("VERIFICACIÓN Y ENTRENAMIENTO DE MODELOS")
print("="*70)

# 1. Verificar imports
print("\n1️⃣ Verificando imports...")
try:
    from config import IMDB_DATASET_PATH, MODELS_DIR, all_models_trained
    print("✅ Config importado")
except Exception as e:
    print(f"❌ Error importando config: {e}")
    sys.exit(1)

# 2. Verificar dataset
print("\n2️⃣ Verificando dataset...")
print(f"   Ruta: {IMDB_DATASET_PATH}")
print(f"   Existe: {IMDB_DATASET_PATH.exists()}")

if not IMDB_DATASET_PATH.exists():
    print("❌ Dataset no encontrado")
    sys.exit(1)

# 3. Cargar datos
print("\n3️⃣ Cargando datos...")
try:
    import pandas as pd
    df = pd.read_csv(IMDB_DATASET_PATH)
    print(f"✅ Cargado: {len(df):,} muestras")
    print(f"   Columnas: {list(df.columns)}")
except Exception as e:
    print(f"❌ Error cargando datos: {e}")
    sys.exit(1)

# 4. Verificar modelos existentes
print("\n4️⃣ Verificando modelos existentes...")
if all_models_trained():
    print("✅ Modelos ya entrenados")
    print("\n✅ PROYECTO VERIFICADO - Todo funcional")
    sys.exit(0)
else:
    print("⚠️  Modelos no encontrados, iniciando entrenamiento...")

# 5. Entrenar modelos
print("\n5️⃣ Entrenando modelos (esto tomará varios minutos)...")
try:
    from train_models import train_all_models_from_file
    results = train_all_models_from_file(str(IMDB_DATASET_PATH), str(MODELS_DIR))
    print("✅ Modelos entrenados exitosamente")
except Exception as e:
    print(f"❌ Error entrenando: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ COMPLETADO - Proyecto listo para usar")
print("="*70)
