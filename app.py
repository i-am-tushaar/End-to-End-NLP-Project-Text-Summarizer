from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import os
import textSummarizer  # IMPORTANT

from textSummarizer.pipeline.prediction import PredictionPipeline


app = FastAPI(title="Text Summarization App")


# --------------------------------------------------
# Locate templates folder dynamically (setup.py safe)
# --------------------------------------------------
PACKAGE_DIR = os.path.dirname(textSummarizer.__file__)
TEMPLATE_PATH = os.path.join(PACKAGE_DIR, "templates")

print("Template Path:", TEMPLATE_PATH)
print("Exists?", os.path.exists(TEMPLATE_PATH))

templates = Jinja2Templates(directory=TEMPLATE_PATH)


# --------------------------------------------------
# Home Page (Custom UI)
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# --------------------------------------------------
# Handle Form Submission
# --------------------------------------------------
@app.post("/", response_class=HTMLResponse)
async def summarize(request: Request, text: str = Form(...)):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(text)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "summary": summary,
                "original_text": text
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": str(e),
                "original_text": text
            }
        )


# --------------------------------------------------
# API Endpoint (Optional JSON API)
# --------------------------------------------------
class TextRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict_api(request: TextRequest):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(request.text)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------
# Training Route
# --------------------------------------------------
@app.get("/train")
async def train():
    try:
        os.system("python main.py")
        return {"message": "Training completed successfully"}
    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------
# Run Server
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
