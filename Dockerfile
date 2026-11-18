FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg fonts-dejavu-core curl unzip && rm -rf /var/lib/apt/lists/*

# Pasang Deno (JavaScript runtime utama rekomendasi resmi yt-dlp)
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -U "yt-dlp[default]" -r requirements.txt

COPY . .
RUN mkdir -p downloads static/clips

CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--timeout", "600", "app:app"]
