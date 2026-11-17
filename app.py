import os
import json
import uuid
import shutil
import subprocess
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

FFMPEG_BIN = shutil.which("ffmpeg") or os.path.abspath("bin/ffmpeg")
DOWNLOAD_DIR = os.path.abspath("downloads")
CLIPS_DIR = os.path.abspath("static/clips")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)


def download_youtube(url, job_id):
    video_path = os.path.join(DOWNLOAD_DIR, f"{job_id}.mp4")
    audio_path = os.path.join(DOWNLOAD_DIR, f"{job_id}.mp3")

    cmd_vid = [
        "yt-dlp",
        "--extractor-args", "youtube:player_client=android,web",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", video_path,
        url
    ]
    subprocess.run(cmd_vid, check=True)

    cmd_aud = [
        FFMPEG_BIN, "-y",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
        audio_path
    ]
    subprocess.run(cmd_aud, check=True)
    return video_path, audio_path


def transcribe_groq(audio_path, groq_key):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {groq_key}"}
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
        data = [
            ("model", "whisper-large-v3"),
            ("language", "id"),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment")
        ]
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=300)
        resp.raise_for_status()
        return resp.json()


def extract_words_safe(transcript_json):
    # Ambil daftar kata jika tersedia
    words = transcript_json.get("words") or []
    if words:
        return words

    # Fallback: pecah per kata dari segments jika Groq hanya mengembalikan segments
    segments = transcript_json.get("segments") or []
    fallback_words = []
    for seg in segments:
        seg_words = seg.get("words") or []
        if seg_words:
            fallback_words.extend(seg_words)
        else:
            tokens = seg.get("text", "").strip().split()
            if not tokens:
                continue
            s_start = float(seg.get("start", 0))
            s_end = float(seg.get("end", s_start + 1))
            step = (s_end - s_start) / len(tokens)
            for i, tok in enumerate(tokens):
                fallback_words.append({
                    "word": tok,
                    "start": s_start + i * step,
                    "end": s_start + (i + 1) * step
                })
    return fallback_words


def find_viral_clips_gemini(transcript_json, gemini_key, num_clips=3):
    segments = transcript_json.get("segments") or []
    formatted_transcript = ""
    for seg in segments:
        start = round(float(seg.get("start", 0)), 1)
        end = round(float(seg.get("end", 0)), 1)
        text = seg.get("text", "").strip()
        formatted_transcript += f"[{start}s - {end}s] {text}\n"

    if not formatted_transcript.strip():
        raise ValueError("Transkrip audio kosong. Pastikan video YouTube memiliki suara ceramah yang jelas.")

    prompt = f"""
Kamu adalah editor ahli konten TikTok dakwah/ceramah ustadz Indonesia.
Analisislah transkrip ceramah berikut beserta timestamp detiknya, lalu pilih {num_clips} potongan (klip) terbaik yang paling menyentuh hati, punya hook kuat di awal, bermakna utuh, dan berpotensi FYP/viral di TikTok.

Aturan ketat:
1. Durasi setiap klip WAJIB antara 30 detik hingga 75 detik.
2. Titik mulai (start_time) dan selesai (end_time) harus pas di awal kalimat dan akhir kalimat yang utuh.
3. Berikan judul hook singkat (maksimal 6 kata) yang menarik perhatian untuk dipasang di bagian atas video.
4. Kembalikan HANYA format JSON valid berupa array objek tanpa markdown ```json atau teks tambahan apa pun.

Format JSON wajib:
[
  {{
    "title": "Judul Hook Singkat",
    "start_time": 120.5,
    "end_time": 175.0,
    "reason": "Alasan singkat kenapa bagian ini menarik"
  }}
]

Transkrip Ceramah:
{formatted_transcript[:28000]}
"""

    url = f"[https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=](https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=){gemini_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4}
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
    if raw_text.endswith("```"):
        raw_text = raw_text.rsplit("\n", 1)[0]

    return json.loads(raw_text.strip())


def format_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def create_ass_subtitle(words, start_clip, end_clip, title, ass_path):
    clip_words = []
    for w in (words or []):
        w_start = float(w.get("start", 0))
        w_end = float(w.get("end", 0))
        if w_end >= start_clip and w_start <= end_clip:
            clip_words.append({
                "word": str(w.get("word", "")).strip(),
                "start": max(0.0, w_start - start_clip),
                "end": max(0.1, w_end - start_clip)
            })

    duration = end_clip - start_clip

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: SubStyle,DejaVu Sans,64,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,60,60,340,1
Style: TitleStyle,DejaVu Sans,58,&H0000FFFF,&H00FFFFFF,&H00000000,&H99000000,-1,0,0,0,100,100,0,0,3,6,0,8,70,70,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 1,0:00:00.00,{format_ass_time(duration)},TitleStyle,,0,0,0,,{title.upper()}
"""

    events = []
    chunk_size = 4
    for i in range(0, len(clip_words), chunk_size):
        chunk = clip_words[i:i + chunk_size]
        for active_idx, active_w in enumerate(chunk):
            ev_start = active_w["start"]
            ev_end = chunk[active_idx + 1]["start"] if active_idx + 1 < len(chunk) else active_w["end"]
            if ev_end <= ev_start:
                ev_end = ev_start + 0.15

            line_parts = []
            for idx, item in enumerate(chunk):
                clean_txt = item["word"].replace("{", "").replace("}", "")
                if idx == active_idx:
                    line_parts.append(f"{{\\c&H00FFFF&}}{clean_txt}{{\\c&HFFFFFF&}}")
                else:
                    line_parts.append(clean_txt)

            full_line = " ".join(line_parts)
            events.append(
                f"Dialogue: 0,{format_ass_time(ev_start)},{format_ass_time(ev_end)},SubStyle,,0,0,0,,{full_line}"
            )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))


def render_clip(video_path, ass_path, start_time, end_time, output_path):
    duration = end_time - start_time
    escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
    vf_filter = f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920,subtitles='{escaped_ass}'"

    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd, check=True)


@app.route("/")
def index():
    has_server_keys = bool(os.environ.get("GROQ_API_KEY") and os.environ.get("GEMINI_API_KEY"))
    return render_template("index.html", has_server_keys=has_server_keys)


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json or {}
    app_pass = os.environ.get("APP_PASSWORD", "")
    if app_pass and data.get("password", "") != app_pass:
        return jsonify({"error": "Password akses salah!"}), 401

    yt_url = data.get("url", "").strip()
    groq_key = data.get("groq_key", "").strip() or os.environ.get("GROQ_API_KEY", "")
    gemini_key = data.get("gemini_key", "").strip() or os.environ.get("GEMINI_API_KEY", "")
    num_clips = int(data.get("num_clips", 3))

    if not yt_url or not groq_key or not gemini_key:
        return jsonify({"error": "Link YouTube dan API Key wajib terisi!"}), 400

    job_id = str(uuid.uuid4())[:8]

    try:
        video_path, audio_path = download_youtube(yt_url, job_id)
        transcript = transcribe_groq(audio_path, groq_key)
        words = extract_words_safe(transcript)
        clips_meta = find_viral_clips_gemini(transcript, gemini_key, num_clips) or []

        results = []
        for idx, clip in enumerate(clips_meta, start=1):
            start_t = float(clip["start_time"])
            end_t = float(clip["end_time"])
            title = clip.get("title", f"Klip Ceramah #{idx}")
            reason = clip.get("reason", "")

            ass_path = os.path.join(DOWNLOAD_DIR, f"{job_id}_clip{idx}.ass")
            out_filename = f"{job_id}_clip{idx}.mp4"
            out_path = os.path.join(CLIPS_DIR, out_filename)

            create_ass_subtitle(words, start_t, end_t, title, ass_path)
            render_clip(video_path, ass_path, start_t, end_t, out_path)

            results.append({
                "title": title,
                "reason": reason,
                "start_time": start_t,
                "end_time": end_t,
                "video_url": f"/static/clips/{out_filename}",
                "filename": out_filename
            })

        return jsonify({"status": "success", "clips": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
