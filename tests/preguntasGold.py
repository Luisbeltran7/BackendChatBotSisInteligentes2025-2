import pandas as pd
import requests

# Rutas de tus archivos
INPUT_CSV = "PreguntasGold.csv"
OUTPUT_CSV = "preguntas_gold_con_respuestas.csv"
BACKEND_URL = "http://localhost:8000/question" # Cambia si usas otro puerto/ruta

def preguntar_al_chatbot(pregunta, model_provider="groq", mode="detallada", top_k=3):
    payload = {
        "question": pregunta,
        "model_provider": model_provider,
        "mode": mode,
        "top_k": top_k,
    }
    try:
        r = requests.post(BACKEND_URL, json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            # Ajusta según el campo que devuelve tu backend
            respuesta = data.get("answer") or data.get("respuesta") or ""
            return respuesta
        else:
            return f"ERROR {r.status_code}: {r.text}"
    except Exception as e:
        return f"ERROR DE CONEXIÓN: {e}"

def main():
    # Lee el archivo, pandas detectará la cabecera
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    print("Columnas detectadas:", df.columns.tolist())

    respuestas = []

    for idx, row in df.iterrows():
        pregunta = row["Pregunta"]
        print(f"Procesando ({idx+1}/{len(df)}): {pregunta!r}")
        resp = preguntar_al_chatbot(pregunta)
        respuestas.append(resp)

    df["respuesta"] = respuestas

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Archivo guardado como: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
