from fastapi import FastAPI, HTTPException
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import pickle
import uuid

app = FastAPI()

class Hyperparameters(BaseModel):
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 1

classification_models = {}

@app.post("/train_model/{model_name}")
async def train_model(model_name: str, hyperparameters: Hyperparameters):
    try:
        X, y = get_training_data(model_name) # Имеется функция, которая получает данные для обучения
        clf = RandomForestClassifier(**hyperparameters.dict())
        clf.fit(X, y)
        classification_models[model_name] = clf
        return {"message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/list_models")
async def list_models():
    return classification_models.keys()

@app.post("/predict/{model_name}")
async def predict(model_name: str, data: list):
    if model_name not in classification_models:
        raise HTTPException(status_code=404, detail="Model not found")
    model = classification_models[model_name]
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}

@app.post("/retrain_model/{model_name}")
async def retrain_model(model_name: str, hyperparameters: Hyperparameters):
    if model_name not in classification_models:
        raise HTTPException(status_code=404, detail="Model not found")
    del classification_models[model_name]
    return await train_model(model_name, hyperparameters)