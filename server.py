from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os, json, requests, re
import torch
import soundfile as sf

app = FastAPI(title="Bamanankan AI Studio Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "generated_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-2-latest")

# --- fallback TTS (local models) ---
try:
    from transformers import VitsModel, AutoTokenizer
    print("Loading local Bambara TTS model...")
    local_model = VitsModel.from_pretrained("sudoping01/bambara-tts")
    local_tokenizer = AutoTokenizer.from_pretrained("sudoping01/bambara-tts")
    HAS_LOCAL_TTS = True
except Exception as e:
    print(f"Local TTS not available: {e}")
    HAS_LOCAL_TTS = False
    local_model = None
    local_tokenizer = None

class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None

class TTSRequest(BaseModel):
    text: str
    filename: str = "bambara_output.wav"

@app.get("/")
def home():
    return {"status": "ok", "service": "Bamanankan AI Studio"}

@app.post("/chat")
def chat(req: ChatRequest):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty")

    if not XAI_API_KEY:
        return {
            "reply": "Grok key is not configured. Set XAI_API_KEY in the environment, then retry.",
            "provider": "fallback"
        }

    system_prompt = req.system_prompt or (
        "Aw ye Bamanankan kɔnɔ baro dɔ be. "
        "Aw bɛ jatew, fɔli, dɛmɛ, ani Bamanankan ka cɛgɛbaga la. "
        "Aw bɛ ka baara la Bamanankan kɔnɔ, ani aw bɛ ka cogo dɔ la ka fɔni kɛ tiɲɛ la."
    )

    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"xAI API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return {"reply": text, "provider": "grok"}
    except Exception as e:
        return {
            "reply": f"Grok API failed: {str(e)}. Using local fallback response.",
            "provider": "fallback"
        }

@app.post("/tts")
def generate_tts(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    filename = req.filename if req.filename.lower().endswith(".wav") else req.filename + ".wav"
    out_path = os.path.join(OUTPUT_DIR, filename)

    if not HAS_LOCAL_TTS:
        # Fallback: create a tiny silent wav to avoid crash if model unavailable.
        import wave
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)
        return {"success": True, "file": f"/audio/{filename}", "message": "Fallback audio generated"}

    try:
        inputs = local_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = local_model(**inputs).waveform
        waveform = output.squeeze().cpu().numpy()
        sr = local_model.config.sampling_rate
        sf.write(out_path, waveform, sr)
        return {"success": True, "file": f"/audio/{filename}", "message": "Audio generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")

app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
