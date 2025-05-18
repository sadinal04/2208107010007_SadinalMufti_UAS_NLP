from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import logging
import uuid
from datetime import datetime
from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("voice-assistant-backend")

# Add file handler for logs
log_dir = os.path.join(os.path.dirname(os.path.abspath(_file_)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"backend_{datetime.now().strftime('%Y%m%d')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

app = FastAPI(title="Voice AI Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/voice-chat")
async def process_voice_chat(request: Request, file: UploadFile = File(...)):
    """
    Process voice input and return an audio response.
    - Accepts a WAV audio file.
    - Transcribes it using Whisper (app/stt).
    - Generates a response using Gemini (app/llm).
    - Converts response to audio using Coqui TTS (app/tts).
    - Returns a WAV audio file.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Processing voice chat with file: {file.filename}")

    try:
        # Validate file
        contents = await file.read()
        file_size = len(contents)
        logger.info(f"[{request_id}] File received: {file_size} bytes")

        if not contents or file_size == 0:
            logger.error(f"[{request_id}] Empty file received")
            return {"error": "Empty file", "request_id": request_id}

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext != ".wav":
            logger.error(f"[{request_id}] Unsupported file extension: {file_ext}")
            return {"error": f"Only WAV files are supported, got {file_ext}", "request_id": request_id}

        # Save input file temporarily
        temp_dir = tempfile.mkdtemp(prefix=f"voice_assistant_{request_id}_")
        input_path = os.path.join(temp_dir, f"input{file_ext}")
        with open(input_path, "wb") as f:
            f.write(contents)
        logger.debug(f"[{request_id}] Saved input file at: {input_path}")

        # Speech-to-Text
        transcript = transcribe_speech_to_text(contents, file_ext=file_ext)
        if isinstance(transcript, str) and transcript.startswith("[ERROR]"):
            logger.error(f"[{request_id}] STT error: {transcript}")
            return {"error": transcript, "request_id": request_id}
        logger.info(f"[{request_id}] Transcribed: {transcript}")

        # Generate response (LLM)
        response = generate_response(transcript)
        if isinstance(response, dict) and "error" in response:
            logger.error(f"[{request_id}] LLM error: {response['error']}")
            return {"error": response['error'], "request_id": request_id}
        g2p_response = response["g2p_response"]
        logger.info(f"[{request_id}] LLM G2P response: {g2p_response}")

        # Text-to-Speech
        output_wav_path = transcribe_text_to_speech(g2p_response)
        if isinstance(output_wav_path, str) and output_wav_path.startswith("[ERROR]"):
            logger.error(f"[{request_id}] TTS error: {output_wav_path}")
            return {"error": output_wav_path, "request_id": request_id}
        if not os.path.exists(output_wav_path):
            logger.error(f"[{request_id}] Output audio file not found at: {output_wav_path}")
            return {"error": "Failed to generate audio", "request_id": request_id}

        logger.info(f"[{request_id}] Audio file generated at: {output_wav_path}")

        # Return audio file
        return FileResponse(
            output_wav_path,
            media_type="audio/wav",
            filename="tts_output.wav"
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}")
        return {"error": f"Internal server error: {str(e)}", "request_id": request_id}
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            # Note: output_wav_path is not deleted here as it’s needed for the response
            # Coqui TTS module should handle its own cleanup if needed
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to clean up files: {str(e)}")

if _name_ == "_main_":
    logger.info("Starting Voice AI Assistant API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)