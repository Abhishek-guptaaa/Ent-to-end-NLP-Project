from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict_pipeline import predict

# Initialize FastAPI app
app = FastAPI()

# Define input data structure
class TextData(BaseModel):
    text: str

@app.post("/predict")
async def get_prediction(data: TextData):
    try:
        predicted_class = predict(data.text)
        return {"text": data.text, "predicted_class": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example endpoint for testing
@app.get("/")
async def root():
    return {"message": "Welcome to the NLP classification API"}
