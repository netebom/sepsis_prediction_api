from typing import Union

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import joblib
from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
import numpy as np
import uvicorn


app = FastAPI(
    title="Sepsis Prediction API",
    description="Predicts whether a patient at the ICU is likely to develop Sepsis or not",
    version="1.0.0",
    contact={
        "name": "Developer",
        "email": "ntuketebom@gmail.com"
    }
)

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.absolute() / "static"),
    name="static"
)
templates = Jinja2Templates(directory='templates')

def load_ml_components():
    models = {
        'KNN': joblib.load(Path(__file__).parent.absolute() / "models" / "Sepsis_knn_model2.pkl"),
        'RF': joblib.load(Path(__file__).parent.absolute() / "models" / 'Sepsis_rf_model2.pkl'),
        'SVM': joblib.load(Path(__file__).parent.absolute() / "models" / 'Sepsis_svm_model2.pkl'),
        'XGB': joblib.load(Path(__file__).parent.absolute() / "models" / 'Sepsis_xgb_model2.pkl'),
    }
    return models


models = load_ml_components()


class SepsisFeatures(BaseModel):
    PRG: float
    PL: float
    PR: float
    SK: float
    TS: float
    M11: float # BMI
    BD2: float  #Blood Work Result-4
    Age: float
    Insurance: int


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post("/predict-single/")
async def search_single(
    request: Request,
    prg: str = Form(...), 
    pl: str = Form(...),
    pr: str = Form(...),
    sk: str = Form(...),
    ts: str = Form(...),
    m11: str = Form(...),
    bd2: str = Form(...),
    age: str = Form(...),
    ins: str = Form(...),
    model: str = Form(...)
    ):
    
    model_type = model
    data: SepsisFeatures = SepsisFeatures(PRG=float(prg), PL=float(pl),PR=float(pr), SK=float(sk), TS=float(ts),M11=float(m11), BD2=float(bd2), Age=float(age), Insurance=int(ins))
    return await process_search(data, model_type)


@app.post("/predict/{model_type}/")
async def api_search(request: Request, data: list[SepsisFeatures], model_type: str):
    _model = models[model_type]
    # data: SepsisFeatures = SepsisFeatures(PRG=float(prg), PL=float(pl),PR=float(pr), SK=float(sk), TS=float(ts),M11=float(m11), BD2=float(bd2), Age=float(age), Insurance=str(ins))
    results = []
    for d in data:
        results.append(await process_search(d, model_type))
    return results


async def process_search(data, model):
    try:
        # Create dataframe from sepsis data
        df = pd.DataFrame([data.model_dump()])
        columns_to_log = ['PRG', 'TS', 'BD2', 'Age']
        df[columns_to_log] = df[columns_to_log].apply(np.log1p)  
        # Call the model model
        prediction_model = models[model]
        # Make prediction
        prediction = prediction_model.predict(df)
        prediction = int(prediction[0])
        prediction_proba = prediction_model.predict_proba(df)[0].tolist() if hasattr(prediction_model, "predict_proba") else [1 - prediction, prediction]
        response = {
            "prediction_model_used": model,
            "prediction": prediction,
            "prediction_probability": {
                "Negative": round(prediction_proba[0], 2),
                "Positive": round(prediction_proba[1], 2)
            }
        }
        return response
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)