from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUIRED_SECTIONS = [
    "Anamnese:",
    "Klinische Untersuchung:",
    "Bildgebung:",
    "Therapie:",
]

SYSTEM_PROMPT = """
You are a medical documentation assistant.

Rewrite the dictated text into correct medical German.

MANDATORY OUTPUT FORMAT — DO NOT CHANGE:

Anamnese:
<text or "nicht diktiert">

Klinische Untersuchung:
<text or "nicht diktiert">

Bildgebung:
<text or "nicht diktiert">

Therapie:
<text or "nicht diktiert">

Rules:
- Every section MUST be present.
- If no information is dictated for a section, write exactly: "nicht diktiert"
- Do not omit sections.
- Do not invent findings.
- Do not add information.
- Use formal medical language suitable for an Arztbrief.
"""

@app.post("/dictate")
async def dictate(audio: UploadFile = File(...)):
    # Save audio temporarily
    suffix = ".webm"
    if audio.filename and "." in audio.filename:
        suffix = "." + audio.filename.rsplit(".", 1)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        audio_path = tmp.name

    try:
        # Transcribe (Whisper)
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="de",
            )
        raw_text = (transcript.text or "").strip()

        # Rewrite into Arztbrief format
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
        )
        result = (resp.choices[0].message.content or "").strip()

        # Enforce mandatory sections
        for sec in REQUIRED_SECTIONS:
            if sec not in result:
                result += f"\n\n{sec}\nnicht diktiert"

        return {"text": result.strip()}

    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass
