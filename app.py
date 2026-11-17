import os
import re
import json
import uuid
import shutil
import traceback
import subprocess
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # Maksimal upload 500MB

FFMPEG_BIN = shutil.which("ffmpeg") or os.path.abspath("bin/ffmpeg")
DOWNLOAD_DIR = os.path.abspath("downloads")
CLIPS_DIR = os.path.abspath("static/clips")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)


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


def find_mp4_url_recursive(obj):
    if isinstance(obj, dict):
        url_val = obj.get("url") or obj.get("link") or obj.get("downloadUrl")
        mime = str(obj.get("mimeType") or obj.get("ext") or obj.get("format") or "").lower()
        has_audio = obj.get("hasAudio", True)
        if isinstance(url_val, str) and url_val.startswith("http"):
            if ("mp4" in mime or "googlevideo.com" in url_val) and has_audio is not False:
                return url_val
        for v in obj.values():
            found = find_mp4_url_recursive(v)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                q = str(item.get("qualityLabel") or item.get("quality") or "")
                u = item.get("url") or item.get("link")
                mime = str(item.get("mimeType") or item.get("ext") or "").lower()
                if isinstance(u, str) and u.startswith("http") and ("720" in q or "480" in q or "360" in q) and "mp4" in mime:
                    return u
        for item in obj:
            found = find_mp4_url_recursive(item)
            if found:
                return found
    return ""


def download_via_rapidapi(yt_url, video_path):
    rapid_key = (os.environ.get("RAPIDAPI_KEY") or "").strip()
    rapid_host = (os.environ.get("RAPIDAPI_HOST") or "").strip()
    rapid_endpoint = (os.environ.get("RAPIDAPI_ENDPOINT") or "").strip()

    if not rapid_key or not rapid_host or not rapid_endpoint:
        return False, "RAPIDAPI_KEY / HOST / ENDPOINT belum diatur."

    vid_id = extract_video_id(yt_url)
    headers = {
        "x-rapidapi-key": rapid_key,
        "x-rapidapi-host": rapid_host
    }

    param_variants = [
        {"id": vid_id} if vid_id else {"url": yt_url},
        {"url": yt_url},
        {"videoId": vid_id} if vid_id else {"url": yt_url}
    ]

    last_err = ""
    for params in param_variants:
        try:
            resp = requests.get(rapid_endpoint, headers=headers, params=params, timeout=30)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:150]}"
                continue

            data = resp.json()
            dl_url = find_mp4_url_recursive(data)
            if not dl_url:
                last_err = f"URL MP4 tidak ditemukan dalam respons RapidAPI: {str(data)[:200]}"
                continue

            with requests.get(dl_url, stream=True, timeout=240, headers={"User-Agent": "Mozilla/5.0"}) as vid_r:
                vid_r.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in vid_r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)

            if os.path.exists(video_path) and os.path.getsize(video_path) > 50000:
                return True, ""
            last_err = "File video terunduh kosong/terlalu kecil."
        except Exception as e:
            last_err = str(e)

    return False, last_err


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
    video_path = os.path.join(DOWNLOAD_DIR, f"{job_id}.mp4")
    audio_path = os.path.join(DOWNLOAD_DIR, f"{job_id}.mp3")

    ok_rapid, rapid_err = download_via_rapidapi(url, video_path)
    if ok_rapid:
        extract_audio(video_path, audio_path)
        return video_path, audio_path

    cookie_args = []
    yt_cookies = (os.environ.get("YOUTUBE_COOKIES") or "").strip()
    if yt_cookies:
        cookie_path = os.path.join(DOWNLOAD_DIR, "cookies.txt")
        with open(cookie_path, "w", encoding="utf-8") as cf:
            cf.write(yt_cookies + "\n")
        cookie_args = ["--cookies", cookie_path]

    cmd_vid = [
        "yt-dlp",
        *cookie_args,
        "--no-playlist",
        "--extractor-args", "youtube:player_client=tv_embedded,android_vr",
        "-f", "best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "-o", video_path,
        url
    ]
    res = subprocess.run(cmd_vid, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(video_path):
        raise RuntimeError(
            f"Gagal unduh YouTube. RapidAPI: ({rapid_err}) | yt-dlp juga diblokir IP datacenter."
        )

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
1. Durasi setiap klip WAJIB antara 25 detik hingga 75 detik.
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

    gemini_host = "generativelanguage.googleapis.com"
    gemini_path = "/v1beta/models/gemini-2.5-flash:generateContent"
    endpoint_url = "https://" + gemini_host + gemini_path + "?key=" + gemini_key

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4}
    }
    resp = requests.post(endpoint_url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini Error ({resp.status_code}): {resp.text[:300]}")

    res_json = resp.json()
    candidates = res_json.get("candidates") if isinstance(res_json, dict) else None
    if not candidates or not isinstance(candidates, list):
        raise RuntimeError(f"Gemini tidak mengembalikan kandidat jawaban: {str(res_json)[:300]}")

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        raise RuntimeError("Gemini mengembalikan parts kosong.")

    raw_text = str(parts[0].get("text") or "").strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
    if raw_text.endswith("```"):
        raw_text = raw_text.rsplit("\n", 1)[0]

    parsed = json.loads(raw_text.strip())
    if not isinstance(parsed, list):
        raise ValueError("Gemini tidak mengembalikan daftar klip yang valid.")
    return [c for c in parsed if isinstance(c, dict)]


def format_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def create_ass_subtitle(words, start_clip, end_clip, title, ass_path):
    safe_words = words if isinstance(words, list) else []
    clip_words = []
    for w in safe_words:
        if not isinstance(w, dict):
            continue
        w_start = float(w.get("start") or 0)
        w_end = float(w.get("end") or 0)
        if w_end >= start_clip and w_start <= end_clip:
            clip_words.append({
                "word": str(w.get("word") or "").strip(),
                "start": max(0.0, w_start - start_clip),
                "end": max(0.1, w_end - start_clip)
            })

    duration = max(1.0, end_clip - start_clip)

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
    duration = max(1.0, end_time - start_time)
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
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg render error: {res.stderr[-300:]}")


@app.route("/")
def index():
    has_server_keys = bool(os.environ.get("GROQ_API_KEY") and os.environ.get("GEMINI_API_KEY"))
    return render_template("index.html", has_server_keys=has_server_keys)


@app.route("/api/generate", methods=["POST"])
def generate():
    if request.content_type and "multipart/form-data" in request.content_type:
        form = request.form
        password = str(form.get("password") or "").strip()
        yt_url = str(form.get("url") or "").strip()
        groq_key = str(form.get("groq_key") or "").strip() or (os.environ.get("GROQ_API_KEY") or "").strip()
        gemini_key = str(form.get("gemini_key") or "").strip() or (os.environ.get("GEMINI_API_KEY") or "").strip()
        num_clips = int(form.get("num_clips") or 3)
        uploaded_file = request.files.get("video_file")
    else:
        data = request.get_json(silent=True) or {}
        password = str(data.get("password") or "").strip()
        yt_url = str(data.get("url") or "").strip()
        groq_key = str(data.get("groq_key") or "").strip() or (os.environ.get("GROQ_API_KEY") or "").strip()
        gemini_key = str(data.get("gemini_key") or "").strip() or (os.environ.get("GEMINI_API_KEY") or "").strip()
        num_clips = int(data.get("num_clips") or 3)
        uploaded_file = None

    app_pass = (os.environ.get("APP_PASSWORD") or "").strip()
    if app_pass and password != app_pass:
        return jsonify({"error": "Password akses salah!"}), 401

    if not groq_key or not gemini_key:
        return jsonify({"error": "GROQ_API_KEY dan GEMINI_API_KEY wajib terisi!"}), 400

    if not yt_url and (not uploaded_file or not uploaded_file.filename):
        return jsonify({"error": "Masukkan Link YouTube ATAU pilih file video MP4!"}), 400

    job_id = str(uuid.uuid4())[:8]

    try:
        if uploaded_file and uploaded_file.filename:
            video_path = os.path.join(DOWNLOAD_DIR, f"{job_id}.mp4")
            audio_path = os.path.join(DOWNLOAD_DIR, f"{job_id}.mp3")
            uploaded_file.save(video_path)
            extract_audio(video_path, audio_path)
        else:
            video_path, audio_path = download_youtube(yt_url, job_id)

        transcript = transcribe_groq(audio_path, groq_key)
        words = extract_words_safe(transcript)
        clips_meta = find_viral_clips_gemini(transcript, gemini_key, num_clips)

        results = []
        for idx, clip in enumerate(clips_meta, start=1):
            start_t = float(clip["start_time"])
            end_t = float(clip["end_time"])
            title = str(clip.get("title") or f"Klip Ceramah #{idx}")
            reason = str(clip.get("reason") or "")

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
        tb_lines = traceback.format_exc().strip().splitlines()
        detail = " | ".join(tb_lines[-3:])
        return jsonify({"error": f"{str(e)} || DETAIL: {detail}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
