from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os, re, joblib

MODEL_PATH = os.getenv("DEPT_MODEL_PATH", "department_pipeline.pkl")
dept_model = joblib.load(MODEL_PATH)

app = FastAPI(title="Booking Feedback Department Classifier API", version="1.0.0")

class FeedbackInput(BaseModel):
    feedback: str

class SentenceResult(BaseModel):
    FeedbackText: str
    Department: str
    DepartmentProb: Optional[float] = None

class ClassifyResponse(BaseModel):
    results: List[SentenceResult]

def preprocess_feedback(text: str) -> str:
    text = re.sub(r'([.!?])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_feedback(text: str) -> List[str]:
    text = preprocess_feedback(text).lower()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    splitter = re.compile(r'\b(?:but|however|although|though|yet)\b', flags=re.IGNORECASE)
    results: List[str] = []

    for sent in sentences:
        parts = splitter.split(sent)
        for part in parts:
            part = part.strip(" ,;:()[]{}\"'")
            if len(part.split()) >= 3:
                results.append(part)

    return results

def max_prob_if_available(model, X: List[str]) -> List[Optional[float]]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return [round(float(p.max()), 3) for p in probs]
    return [None] * len(X)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify", response_model=ClassifyResponse)
def classify(input: FeedbackInput):
    fb = (input.feedback or "").strip()
    if not fb:
        raise HTTPException(status_code=400, detail="Feedback is empty")

    parts = split_feedback(fb)
    if not parts:
        parts = [preprocess_feedback(fb).lower().strip()]

    preds = dept_model.predict(parts).tolist()
    probs = max_prob_if_available(dept_model, parts)

    results = []
    for i, t in enumerate(parts):
        results.append({
            "FeedbackText": t,
            "Department": preds[i],
            "DepartmentProb": probs[i]
        })

    return {"results": results}