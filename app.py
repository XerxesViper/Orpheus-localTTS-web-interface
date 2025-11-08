import os
import re
import requests
import trafilatura
import threading
import subprocess
from time import perf_counter
from flask import Flask, render_template, request, jsonify
from gguf_orpheus import generate_speech_from_api

import logging
from io import StringIO

# Setup logging
log_capture = StringIO()
handler = logging.StreamHandler(log_capture)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Store logs globally
app_logs = []


def log_message(message):
    """Add message to logs"""
    app_logs.append(message)
    print(message)  # Also print to console


# CONFIG - Change this to your Google Drive folder later
OUTPUT_BASE_DIR = r"E:\Article TTS"

app = Flask(__name__)

# Track progress globally (simple, works for single user)
progress_tracker = {}


def split_text_into_chunks(text, hard_limit=150):
    """
    Split text by sentences.
    - Aim for ~100 words per chunk (soft limit)
    - Never exceed 150 words per chunk (hard limit)
    - If a single sentence > 150 words, split it
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if sentence_words > hard_limit:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                word_count = 0

            words = sentence.split()
            for i in range(0, len(words), hard_limit):
                chunks.append(' '.join(words[i:i + hard_limit]))

        elif word_count + sentence_words > hard_limit:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            word_count = sentence_words

        else:
            current_chunk.append(sentence)
            word_count += sentence_words

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def fetch_article(url):
    """Download and extract article from URL."""
    if "404media" in url:
        session = requests.Session()
        cookies = {
            'ghost-members-ssr': '3e3dd2ee-1026-4606-91d4-0ed59dd47328',
            'ghost-members-ssr.sig': 'kmki5yGY3xS53M3hlUVPeRp-LqY'
        }
        response = requests.get(url, cookies=cookies)
        content = response.content
    else:
        content = trafilatura.fetch_url(url)

    text = trafilatura.extract(content)
    metadata = trafilatura.extract_metadata(content)

    return text, metadata


@app.route('/fetch_article', methods=['POST'])
def fetch_article_route():
    data = request.json
    url = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        article, metadata = fetch_article(url)

        if not article:
            return jsonify({'error': 'Failed to extract article content'}), 400

        return jsonify({
            'title': metadata.title,
            'content': article
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def convert_article_to_audio(title, content, hard_limit, article_id):
    """Convert article to audio - runs in background thread."""
    try:
        app_logs.clear()  # Clear logs for new conversion
        log_message(f"[START] Converting: {title}")

        progress_tracker[article_id] = {
            'status': 'converting',
            'current': 0,
            'total': 0,
            'error': None
        }

        # Sanitize title for folder name
        safe_title = re.sub(r'[^\w\s-]', '', title)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        log_message(f"[INFO] Safe title: {safe_title}")

        # Create folder for this article
        article_output_dir = os.path.join(OUTPUT_BASE_DIR, safe_title)

        if os.path.exists(article_output_dir):
            import shutil
            log_message(f"[INFO] Clearing existing folder: {article_output_dir}")
            shutil.rmtree(article_output_dir)

        os.makedirs(article_output_dir, exist_ok=True)
        log_message(f"[INFO] Created output folder: {article_output_dir}")

        # Clean content
        article = content.strip()
        article = re.sub(r'\s+', ' ', article)
        article = article.replace('"', '"').replace('"', '"')
        article = article.replace(''', "'").replace(''', "'")

        # Split into chunks
        chunks = split_text_into_chunks(article, hard_limit=hard_limit)
        progress_tracker[article_id]['total'] = len(chunks)
        log_message(f"[INFO] Split into {len(chunks)} chunks")

        # Generate audio for each chunk
        chunk_files = []
        for i, chunk in enumerate(chunks):
            output_file = os.path.join(
                article_output_dir,
                f"{safe_title}_{i + 1:03d}.wav"
            )
            log_message(f"[CHUNK {i + 1}/{len(chunks)}] Generating audio...")
            generate_speech_from_api(chunk, voice="tara", output_file=output_file)
            chunk_files.append(output_file)
            progress_tracker[article_id]['current'] = i + 1
            log_message(f"[CHUNK {i + 1}/{len(chunks)}] ✓ Completed")

        # Combine WAV files into MP3
        log_message(f"[INFO] Combining {len(chunks)} chunks into MP3...")
        combine_audio_files(chunk_files, article_output_dir, safe_title)
        log_message(f"[SUCCESS] MP3 created: {safe_title}.mp3")

        progress_tracker[article_id]['status'] = 'completed'
        log_message(f"[END] Conversion completed successfully")

    except Exception as e:
        progress_tracker[article_id]['error'] = str(e)
        log_message(f"[ERROR] {str(e)}")
        print(f"Error: {e}")


def combine_audio_files(wav_files, output_dir, title):
    """Combine multiple WAV files into a single MP3 using ffmpeg."""
    try:
        # Create a text file with list of files for ffmpeg
        concat_file = os.path.join(output_dir, 'concat_list.txt')
        with open(concat_file, 'w') as f:
            for wav_file in wav_files:
                f.write(f"file '{wav_file}'\n")

        # Output MP3 path
        output_mp3 = os.path.join(output_dir, f"{title}.mp3")

        # Run ffmpeg command
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c:a', 'libmp3lame',
            '-q:a', '4',
            '-y',
            output_mp3
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Combined MP3 saved: {output_mp3}")

        # Clean up concat file
        os.remove(concat_file)

    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
    except Exception as e:
        print(f"Error combining audio: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/logs/<article_id>')
def get_logs(article_id):
    """Return all logs for this conversion"""
    return jsonify({'logs': app_logs})


@app.route('/convert', methods=['POST'])
def convert():
    data = request.json
    url = data.get('url', '').strip()
    hard_limit = int(data.get('hard_limit', 80))

    # Support both fetched URL and manual text input
    title = data.get('title', '')
    content = data.get('content', '')

    if not title or not content:
        if not url:
            return jsonify({'error': 'Either URL or title+content required'}), 400

        # Fetch from URL
        try:
            article, metadata = fetch_article(url)
            if not article:
                return jsonify({'error': 'Failed to extract article'}), 400
            title = metadata.title
            content = article
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    article_id = f"convert_{int(perf_counter() * 1000)}"
    thread = threading.Thread(
        target=convert_article_to_audio,
        args=(title, content, hard_limit, article_id)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'article_id': article_id})


@app.route('/progress/<article_id>')
def get_progress(article_id):
    if article_id not in progress_tracker:
        return jsonify({'error': 'Invalid article ID'}), 404

    return jsonify(progress_tracker[article_id])


if __name__ == '__main__':
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    app.run(debug=False, host='0.0.0.0', port=5000)
