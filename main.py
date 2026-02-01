from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import traceback

from google import genai

# --- Config ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # good default
# Gemini client reads key from env var GEMINI_API_KEY (recommended by Google docs)
client = genai.Client()

app = FastAPI()

# CORS: allow your GitHub Pages frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # later you can restrict to your Pages domain
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)

REQUIRED_SECTIONS = [
    "Anamnese:",
    "Klinische Untersuchung:",
    "Bildgebung:",
    "Diagnose:",
    "Therapie:",
]

SYSTEM_PROMPT = """
You are a medical documentation assistant.

Task:
Rewrite the dictated German text into correct medical German and format it into exactly these sections:

Anamnese:
Klinische Untersuchung:
Bildgebung:
Diagnose:
Therapie:

Rules:
- Every section MUST be present.
- If a section was not dictated, write exactly: "nicht diktiert"
- Do not omit sections.
- Do not invent findings.
- Do not add information.
- Use formal medical language suitable for an Arztbrief.
"""

class FormatRequest(BaseModel):
    text: str

def ensure_all_sections(text: str) -> str:
    out = (text or "").strip()
    for sec in REQUIRED_SECTIONS:
        if sec not in out:
            out += f"\n\n{sec}\nnicht diktiert"
    return out.strip()

@app.get("/")
def root():
    return {"status": "ok", "service": "medical-dictation-gemini"}

@app.get("/health")
def health():
    # helpful sanity check
    return {
        "ok": True,
        "has_gemini_key": bool(os.getenv("GEMINI_API_KEY")),
        "model": GEMINI_MODEL,
    }

@app.post("/format")
def format_brief(req: FormatRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text.")

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing. Add it in Railway → Variables.",
        )

    try:
        prompt = f"{SYSTEM_PROMPT}\n\nDictation:\n{req.text.strip()}\n"
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        result = (resp.text or "").strip()
        result = ensure_all_sections(result)
        return {"text": result}
    except Exception as e:
        print("GEMINI ERROR:", repr(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=502, detail=f"Gemini failed: {str(e)[:400]}")
