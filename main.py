

import sys
import joblib
import pandas as pd
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


MODEL_PATH = Path("Logistic Regression_model_2112.pkl")

CAUSA_MAPPING = {
    "none": 0,
    "cardiopatias_congenitas": 1,
    "evop_hcp": 2,
    "enf_tejido_conectivo": 3,
    "infeccion_vih": 4,
    "colangitis_biliar": 5,
    "idiopatica": 6,
    "asociada_drogas_toxinas": 7,
    "hereditaria": 8,
    "esquistosomiasis": 9,
    "hipertension_portal": 10,
    "hp_neonato": 11,
    "en_estudio": 12
}


def load_pipeline(model_path: Path):
    """Carga el pipeline de ML de forma segura."""
    if not model_path.exists():
        print(f"❌ Error: Archivo del modelo no encontrado en {model_path.resolve()}")
        print("Asegúrate de que 'Logistic Regression_model_2112.pkl' esté en la misma carpeta que este script.")
        sys.exit(1)
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        sys.exit(1)


def extract_feature_list(pipeline):
    """Extrae los nombres de las features del pipeline."""
    if hasattr(pipeline, 'feature_names_in_'):
        return list(pipeline.feature_names_in_)
    if 'scaler' in pipeline.named_steps and hasattr(pipeline.named_steps['scaler'], 'feature_names_in_'):
        return list(pipeline.named_steps['scaler'].feature_names_in_)
    print("⚠ Advertencia: No se pudieron extraer los nombres de las features.")
    return []


def pad_dataframe(df_partial, full_features, pipeline):
    """Rellena las features faltantes con la media del scaler o ceros."""
    row_data = {}
    scaler = pipeline.named_steps.get('scaler', None)
    
    if scaler is not None and hasattr(scaler, 'mean_'):
        for col, mean_val in zip(scaler.feature_names_in_, scaler.mean_):
            row_data[col] = mean_val
    else:
        print("⚠ Advertencia: No se encontraron medias del scaler. Rellenando con ceros.")
        row_data = {col: 0.0 for col in full_features}

    for col in df_partial.columns:
        if col in row_data:
            row_data[col] = df_partial[col].iloc[0]

    for feature in full_features:
        row_data.setdefault(feature, 0.0)

    return pd.DataFrame([row_data], columns=full_features)


def map_prediction(pred):
    """Mapea la salida del modelo a la etiqueta."""
    return (
        "Sin sospecha de Hipertensión Pulmonar"
        if pred == 3 else
        "Sospecha de Hipertensión Pulmonar"
    )


class PredictionInput(BaseModel):
    fatiga: int
    bendopnea: int
    hemoptisis: int
    sincope: int
    causa_seleccionada: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Cargando modelo...")
pipeline = load_pipeline(MODEL_PATH)
features = extract_feature_list(pipeline)
if not features:
    print("❌ Error fatal: No se pudo determinar la lista de features del modelo.")
    sys.exit(1)
print(f"✅ Modelo cargado y listo desde: {MODEL_PATH.resolve()}")


@app.post("/predict")
async def predict(data: PredictionInput):
    """Recibe datos del frontend y devuelve una predicción."""
    try:
        causa_g1 = CAUSA_MAPPING.get(data.causa_seleccionada)
        if causa_g1 is None:
            raise ValueError(f"Valor 'causa_seleccionada' inválido: {data.causa_seleccionada}")

        X_partial = pd.DataFrame([{
            'causa_g1': causa_g1,
            'fatiga': data.fatiga,
            'bendopnea': data.bendopnea,
            'hemoptisis': data.hemoptisis,
            'sincope': data.sincope
        }])

        X_ready = pad_dataframe(X_partial, features, pipeline)
        pred = pipeline.predict(X_ready)[0]
        label = map_prediction(pred)

        return {"success": True, "diagnosis": label}
    
    except Exception as e:
        print(f"❌ Error durante la predicción: {e}")
        raise HTTPException(
            status_code=400, 
            detail={"success": False, "detail": str(e)}
        )

 
if __name__ == "__main__":
    print(f'\n{"="*35} 🩺 Iniciando Servidor API de HP 🫁 {"="*35}\n')
    uvicorn.run(app, host="0.0.0.0", port=8000)