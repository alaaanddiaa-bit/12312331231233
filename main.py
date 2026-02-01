from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/dictate")
async def dictate(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    transcript = openai.audio.transcriptions.create(
        file=open(tmp_path, "rb"),
        model="gpt-4o-transcribe"
    )

    prompt = f"""
Erstelle einen medizinischen Arztbrief mit folgenden Abschnitten
(alle müssen vorhanden sein, sonst 'nicht diktiert'):

Anamnese
Klinische Untersuchung
Bildgebung
Therapie

Text:
{transcript.text}
"""

    completion = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"text": completion.choices[0].message.content}
