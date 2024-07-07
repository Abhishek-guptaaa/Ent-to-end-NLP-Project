from hate.pipeline.train import TrainPipeline
from fastapi import FastAPI, HTTPException
import uvicorn
import sys
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from hate.pipeline.predict import PredictionPipeline
from hate.exception import CustomException
from config.config import Config

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict")
async def predict_route(text: str):
    try:
        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.predict(text)
        return {"text": text, "prediction": prediction}
    except Exception as e:
        raise CustomException(e, sys) from e

if __name__ == "__main__":
    uvicorn.run(app, host=Config.APP_HOST, port=Config.APP_PORT)
