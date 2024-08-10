from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
from starlette.responses import FileResponse
app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Specify the absolute path for the model directory
output_dir = "Static/HumanAIModel"  # Change this to the actual path
tokenizer = BertTokenizer.from_pretrained(output_dir)
model = BertForSequenceClassification.from_pretrained(output_dir)

# Define request body model
class TextData(BaseModel):
    text: str

# Define a function to make predictions on new text
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_label = "AI" if predicted_class == 0 else "HUMAN"  # Assuming 'BO' is label 0 and 'EM' is label 1
    return predicted_label

# Prediction endpoint
@app.post('/predict')
async def predict_endpoint(data: TextData):
    text = data.text
    predicted_label = predict(text)
    return {'predicted_label': predicted_label}

# Serve HTML file
@app.get('/')
async def serve_index():
    return FileResponse("Static/index.html")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)