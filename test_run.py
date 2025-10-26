import os
from src import model

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(PROJECT_ROOT, "IMDB Dataset.csv")
model_path = os.path.join(PROJECT_ROOT, "models", "review_model.joblib")

if not os.path.exists(csv_path):
    print(f"CSV no encontrado: {csv_path}")
    raise SystemExit(1)

print("Entrenando modelo (esto puede tardar tens de segundos/minutos según tamaño del CSV)...")
model.train_from_csv(csv_path, model_path=model_path, n_clusters=50)
print("Entrenamiento completado. Pruebas rápidas:")

tests = [
    ("Vi esta película anoche y salí encantado. La dirección es brillante, la banda sonora acompaña cada escena y las actuaciones son impecables.", "reseña"),
    ("El director anunció hoy en rueda de prensa las nuevas fechas de rodaje y el reparto confirmado para la próxima película.", "no-reseña"),
    ("Ingredientes: 2 tazas de harina, 1 huevo. Preparación: mezclar y hornear.", "no-reseña"),
    ("Buena trama, actuaciones correctas, entretenida.", "reseña"),
    ("La película promete mucho pero se queda en nada. El guion es pobre y no la recomendaría.", "reseña"),
]

for txt, expected in tests:
    res = model.predict_text(txt, model_path=model_path)
    decision = "reseña" if res["is_review"] else "no-reseña"
    print("---")
    print("Texto:", txt[:80] + "..." if len(txt) > 80 else txt)
    print("Esperado:", expected, "| Predicho:", decision)
    print(f"Prob: {res['probability']:.3f} | Sim: {res['similarity']:.3f} | KW: {res['keyword_score']:.3f} | Eval: {res['eval_score']:.3f} | FP: {res['first_person_score']:.3f}")
    if res["matched_evaluative"]:
        print("Eval words:", ", ".join(res["matched_evaluative"]))
    if res["matched_first_person"]:
        print("First person:", ", ".join(res["matched_first_person"]))
