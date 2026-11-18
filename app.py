import os
import re
import json
import uuid
import base64
import shutil
import traceback
import subprocess
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

FFMPEG_BIN = shutil.which("ffmpeg") or os.path.abspath("bin/ffmpeg")
DOWNLOAD_DIR = os.path.abspath("downloads")
CLIPS_DIR = os.path.abspath("static/clips")
COOKIES_PATH = os.path.abspath("youtube_cookies.txt")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)


def prepare_cookies_file():
    raw_cookies = (os.environ.get("YOUTUBE_COOKIES") or "").strip()
    if not raw_cookies:
        return None
    if not raw_cookies.startswith("#") and ".youtube.com" not in raw_cookies[:100]:
        try:
            decoded = base64.b64decode(raw_cookies).decode("utf-8", errors="ignore")
            if ".youtube.com" in decoded:
                raw_cookies = decoded
        except Exception:
            pass
    with open(COOKIES_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(raw_cookies.strip() + "\n")
    return COOKIES_PATH


def extract_video_id(yt_url):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:shorts\/)([0-9A-Za-z_-]{11})"
    ]
    for pat in patterns:
        m = re.search(pat, yt_url)
        if m:
            return m.group(1)
    return ""


def extract_audio(video_path, audio_path):
    cmd_aud = [
        FFMPEG_BIN, "-y",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
        audio_path
    ]
    res_aud = subprocess.run(cmd_aud, capture_output=True, text=True)
    if res_aud.returncode != 0:
        raise RuntimeError(f"FFmpeg audio gagal: {res_aud.stderr[-300:]}")


def download_youtube(url, job_id):
    vid_id = extract_video_id(url)
    if not vid_id:
        raise RuntimeError("Link YouTube tidak valid. Pastikan formatnya https://www.youtube.com/watch?v=...")

    clean_url = f"https://www.youtube.com/watch?v={vid_id}"
    video_path = os.path.join(DOWNLOAD_DIR, f"{job_id}.mp4")
    audio_path = os.path.join(DOWNLOAD_DIR, f"{job_id}.mp3")

    cookie_file = prepare_cookies_file()
    if not cookie_file:
        raise RuntimeError("Variable YOUTUBE_COOKIES belum diisi di Railway!")

    cmd_vid = [
        "yt-dlp",
        "--no-playlist",
        "--cookies", cookie_file,
        "-f", "bv*[vcodec^=avc1][height<=720][ext=mp4]+ba[ext=m4a]/bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format", "mp4",
        "-o", video_path,
        clean_url
    ]

    res = subprocess.run(cmd_vid, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(video_path) or os.path.getsize(video_path) < 100000:
        err_tail = (res.stderr or res.stdout or "Unknown yt-dlp error").strip()[-450:]
        raise RuntimeError(f"yt-dlp gagal mengunduh video: {err_tail}")

    extract_audio(video_path, audio_path)
    return video_path, audio_path


def transcribe_groq(audio_path, groq_key):
    base_groq = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {groq_key}"}
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
        data = {
            "model": "whisper-large-v3",
            "language": "id",
            "response_format": "verbose_json"
        }
        resp = requests.post(base_groq, headers=headers, files=files, data=data, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"Groq Error ({resp.status_code}): {resp.text[:300]}")
        parsed = resp.json()
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Groq mengembalikan format tidak terduga: {type(parsed)}")
        return parsed


def extract_words_safe(transcript_json):
    if not isinstance(transcript_json, dict):
        return []

    raw_words = transcript_json.get("words")
    if isinstance(raw_words, list) and len(raw_words) > 0:
        return [w for w in raw_words if isinstance(w, dict)]

    segments = transcript_json.get("segments")
    if not isinstance(segments, list):
        return []

    fallback_words = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_words = seg.get("words")
        if isinstance(seg_words, list) and len(seg_words) > 0:
            fallback_words.extend([w for w in seg_words if isinstance(w, dict)])
        else:
            tokens = str(seg.get("text") or "").strip().split()
            if not tokens:
                continue
            s_start = float(seg.get("start") or 0)
            s_end = float(seg.get("end") or (s_start + 1))
            step = max(0.05, (s_end - s_start) / len(tokens))
            for i, tok in enumerate(tokens):
                fallback_words.append({
                    "word": tok,
                    "start": s_start + i * step,
                    "end": s_start + (i + 1) * step
                })
    return fallback_words


def get_available_gemini_models(gemini_key):
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}"
    try:
        r = requests.get(list_url, timeout=20)
        if r.status_code == 200:
            models_data = r.json().get("models") or []
            valid_flash = []
            valid_others = []
            for m in models_data:
                methods = m.get("supportedGenerationMethods") or []
                name = str(m.get("name") or "")
                if "generateContent" in methods and "gemini" in name.lower():
                    short_name = name.replace("models/", "")
                    if "flash" in short_name.lower() and "image" not in short_name.lower() and "audio" not in short_name.lower():
                        valid_flash.append(short_name)
                    else:
                        valid_others.append(short_name)
            valid_flash.sort(reverse=True)
            combined = valid_flash + valid_others
            if combined:
                return combined
    except Exception:
        pass
    return ["gemini-3.8-flash", "gemini-3.0-flash", "gemini-2.5-flash"]


def find_viral_clips_gemini(transcript_json, gemini_key, num_clips=3):
    segments = transcript_json.get("segments") if isinstance(transcript_json, dict) else []
    if not isinstance(segments, list):
        segments = []

    formatted_transcript = ""
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = round(float(seg.get("start") or 0), 1)
        end = round(float(seg.get("end") or 0), 1)
        text = str(seg.get("text") or "").strip()
        formatted_transcript += f"[{start}s - {end}s] {text}\n"

    if not formatted_transcript.strip():
        raw_text_full = str(transcript_json.get("text") or "").strip()
        if raw_text_full:
            formatted_transcript = f"[0.0s - 60.0s] {raw_text_full}\n"
        else:
            raise ValueError("Transkrip audio dari Groq kosong.")

    prompt = f"""
Kamu adalah editor ahli konten TikTok dakwah/ceramah ustadz Indonesia.
Analisislah transkrip ceramah berikut beserta timestamp detiknya, lalu pilih {num_clips} potongan (klip) terbaik yang paling menyentuh hati, punya hook kuat di awal, bermakna utuh, dan berpotensi FYP/viral di TikTok.

Aturan ketat:
1. Durasi setiap klip WAJIB antara 60 detik hingga 90 detik.
2. Titik mulai (start_time) dan selesai (end_time) harus pas di awal kalimat dan akhir kalimat yang utuh.
3. Berikan judul hook singkat (maksimal 6 kata) yang menarik perhatian untuk dipasang di bagian atas video.
4. Kembalikan HANYA format JSON valid berupa array objek tanpa markdown ```json atau teks tambahan apa pun.

Format JSON wajib:
[
  {{
    "title": "Judul Hook Singkat",
    "start_time": 120.5,
    "end_time": 195.0,
    "reason": "Alasan singkat kenapa bagian ini menarik"
  }}
]

Transkrip Ceramah:
{formatted_transcript[:28000]}
"""

    gemini_host = "generativelanguage.googleapis.com"
    models_to_try = get_available_gemini_models(gemini_key)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4}
    }

    res_json = None
    errors_log = []
    for model_name in models_to_try[:5]:
        endpoint_url = f"
