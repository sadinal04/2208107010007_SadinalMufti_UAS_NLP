from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app import stt, llm, tts
import os
import tempfile

app = FastAPI()

# Izinkan CORS agar Gradio bisa mengakses API lokal
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):
    # Baca file audio
    audio_bytes = await file.read()

    # STEP 1: Transkripsi audio (STT)
    transcript = stt.transcribe_speech_to_text(audio_bytes)

    if "[ERROR]" in transcript:
        return {"error": "STT gagal: " + transcript}

    # STEP 2: Kirim ke Gemini LLM
    response_text = llm.generate_response(transcript)

    if "[ERROR]" in response_text:
        return {"error": "LLM gagal: " + response_text}

    # STEP 3: Konversi ke audio (TTS)
    tts_path = tts.transcribe_text_to_speech(response_text)

    if "[ERROR]" in tts_path:
        return {"error": "TTS gagal: " + tts_path}

    # STEP 4: Kirimkan kembali audio hasil TTS ke Gradio
    return FileResponse(tts_path, media_type="audio/wav")
