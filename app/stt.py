import os
import uuid
import tempfile
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# path ke folder utilitas STT
WHISPER_DIR = os.path.join(BASE_DIR, "whisper.cpp")

# TODO: Lengkapi path ke binary whisper-cli
# Gunakan os.path.join() untuk menggabungkan WHISPER_DIR, "build", "bin", dan "whisper-cli"
WHISPER_BINARY = os.path.join(WHISPER_DIR, "main")

# TODO: Lengkapi path ke file model Whisper (contoh: ggml-large-v3-turbo.bin)
# Gunakan os.path.join() untuk mengarah ke file model di dalam folder "models"
WHISPER_MODEL_PATH = os.path.join(WHISPER_DIR, "models", "ggml-large-v3-turbo.bin")

def transcribe_speech_to_text(file_bytes: bytes, file_ext: str = ".wav") -> str:
    """
    Transkrip file audio menggunakan whisper.cpp CLI
    Args:
        file_bytes (bytes): Isi file audio
        file_ext (str): Ekstensi file, default ".wav"
    Returns:
        str: Teks hasil transkripsi
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, f"{uuid.uuid4()}{file_ext}")
        result_path = os.path.join(tmpdir, "transcription.txt")

        # simpan audio ke file temporer
        with open(audio_path, "wb") as f:
            f.write(file_bytes)

        # jalankan whisper.cpp dengan subprocess
        cmd = [
            WHISPER_BINARY,
            "-m", WHISPER_MODEL_PATH,
            "-f", audio_path,
            "-otxt",
            "-of", os.path.join(tmpdir, "..", "transcription")
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            return f"[ERROR] Whisper failed: {e}"
        
        # baca hasil transkripsi
        try:
            with open(result_path, "r", encoding="utf-8") as result_file:
                return result_file.read()
        except FileNotFoundError:
            return "[ERROR] Transcription file not found"
