import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load variabel lingkungan dari file .env
load_dotenv()

# Ambil API key dari environment
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Konstanta
MODEL = "gemini-2.0-flash"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # __file__, bukan file
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

# Instruksi sistem
system_instruction = """ 
You are a responsive, intelligent, and fluent virtual assistant who communicates in Indonesian. 
Your task is to provide clear, concise, and informative answers in response to user queries or statements spoken through voice. 

Your answers must: 
- Be written in polite and easily understandable Indonesian. 
- Be short and to the point (maximum 2–3 sentences). 
- Avoid repeating the user's question; respond directly with the answer. 

Example tone: 
User: Cuaca hari ini gimana? 
Assistant: Hari ini cuacanya cerah di sebagian besar wilayah, dengan suhu sekitar 30 derajat. 

User: Kamu tahu siapa presiden Indonesia? 
Assistant: Presiden Indonesia saat ini adalah Joko Widodo. 

If you're unsure about an answer, be honest and say that you don't know. 
"""

# Konfigurasi Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Inisialisasi model
model = genai.GenerativeModel(model_name=MODEL)

# Fungsi menyimpan riwayat chat
def save_chat_history(chat):
    try:
        history = chat.history
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([msg.to_dict() for msg in history], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan riwayat chat: {e}")

# Fungsi memuat riwayat chat
def load_chat_history():
    try:
        if os.path.exists(CHAT_HISTORY_FILE) and os.path.getsize(CHAT_HISTORY_FILE) > 0:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history_json = json.load(f)
            history = [genai.types.Content.from_dict(msg) for msg in history_json]
            return model.start_chat(history=history)
    except Exception as e:
        print(f"[ERROR] Gagal memuat riwayat chat: {e}")
    return model.start_chat()

# Mulai sesi chat
chat = load_chat_history()

# Fungsi untuk menghasilkan respons
def generate_response(prompt: str) -> str:
    try:
        if not chat.history:
            chat.send_message(system_instruction)
        response = chat.send_message(prompt)
        save_chat_history(chat)
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] {e}"
