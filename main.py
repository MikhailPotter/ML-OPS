from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import dump, load
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import List, Optional
import os

app = FastAPI()

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

class Data(BaseModel):
    features: List[float]
    target: int

class ModelParams(BaseModel):
    model_name: str
    model_type: str
    hyperparameters: Optional[dict] = None

class PredictParams(BaseModel):
    model_name: str
    features: List[float]

@app.post("/train/")
def train_model(params: ModelParams, data: List[Data]):
    # Создание и обучение модели
    features = np.array([item.features for item in data])
    target = np.array([item.target for item in data])

    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)

    if params.model_type == "LogisticRegression":
        model = LogisticRegression(**params.hyperparameters)
    elif params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(**params.hyperparameters)
    else:
        raise HTTPException(status_code=400, detail="Unsupported model type")

    model.fit(features_train, target_train)

    # Оценка качества модели
    target_pred = model.predict(features_test)
    accuracy = accuracy_score(target_test, target_pred)

    # Сохранение модели
    model_path = os.path.join(MODELS_DIR, params.model_name)
    dump(model, model_path)

    return {"model_name": params.model_name, "accuracy": accuracy}

@app.post("/predict/")
def predict(params: PredictParams):
    model_path = os.path.join(MODELS_DIR, params.model_name)
    model = load(model_path)

    features = np.array(params.features)
    prediction = model.predict(features.reshape(1, -1))

    return {"prediction": prediction[0]}

@app.get("/models/")
def list_models():
    model_names = [model for model in os.listdir(MODELS_DIR) if model.endswith(".joblib")]
    return {"models": model_names}

@app.delete("/models/{model_name}")
def delete_model(model_name: str):
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        os.remove(model_path)
        return {"model_name": model_name, "status": "deleted"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")