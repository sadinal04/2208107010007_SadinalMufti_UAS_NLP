import os
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile

def voice_chat(audio):
    if audio is None:
        return None
    
    sr, audio_data = audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
        audio_path = tmpfile.name

    with open(audio_path, "rb") as f:
        files = {"file": ("voice.wav", f, "audio/wav")}
        response = requests.post("http://localhost:8000/voice-chat", files=files)

    if response.status_code == 200:
        output_audio_path = os.path.join(tempfile.gettempdir(), "tts_output.wav")
        with open(output_audio_path, "wb") as f:
            f.write(response.content)
        return output_audio_path
    else:
        return None

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("## 🎙️ **Voice Assistant AI**")
    gr.Markdown("Interaktif, real-time, dan responsif! Bicara langsung ke mikrofon dan dengarkan balasan dari AI asisten pribadi Anda.")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🎤 Input Suara Anda")
                audio_input = gr.Audio(sources="microphone", type="numpy", label="Klik tombol untuk mulai merekam:")
                submit_btn = gr.Button("🔁 Kirim Pertanyaan")
                gr.Markdown("⚠️ Pastikan mikrofon aktif dan izinkan browser untuk mengaksesnya.")

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🔊 Balasan Suara AI")
                audio_output = gr.Audio(type="filepath", label="Hasil dari Asisten AI")

    submit_btn.click(fn=voice_chat, inputs=audio_input, outputs=audio_output)

    with gr.Accordion("ℹ️ Cara Menggunakan", open=False):
        gr.Markdown("""
        1. Klik ikon mikrofon dan sampaikan pertanyaan Anda.
        2. Tekan tombol **Submit** untuk mengirim suara ke AI.
        3. Dengarkan jawaban yang diberikan secara otomatis.
        4. Jika tidak ada suara, periksa izin mikrofon dan koneksi server.
        """)

demo.launch()
