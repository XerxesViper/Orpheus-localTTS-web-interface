import os
import re
import requests
import trafilatura
import threading
import subprocess
from time import perf_counter
from flask import Flask, render_template, request, jsonify
from gguf_orpheus import generate_speech_from_api
from queue import Queue
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

OUTPUT_BASE_DIR = r"E:\Article TTS"

app = Flask(__name__)

# ========== QUEUE SYSTEM ==========
job_counter = 0
job_counter_lock = threading.Lock()

job_queue = Queue(maxsize=10)  # Max 10 jobs
current_job = None
current_job_lock = threading.Lock()
completed_jobs = []  # Last 5 completed
completed_jobs_lock = threading.Lock()

# Track logs and progress for current job
progress_tracker = {}
app_logs = {}


def get_next_job_id():
    """Get next sequential job ID"""
    global job_counter
    with job_counter_lock:
        job_counter += 1
        return job_counter


def log_message(job_id, message):
    """Add message to job's logs"""
    if job_id not in app_logs:
        app_logs[job_id] = []
    app_logs[job_id].append(message)
    print(message)


def split_text_into_chunks(text, hard_limit=150):
    """Split text by sentences"""
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
    """Download and extract article from URL"""
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


def convert_article_to_audio(job):
    """Convert article to audio - runs in background thread"""
    job_id = job['job_id']
    title = job['title']
    content = job['content']
    hard_limit = job['hard_limit']

    try:
        app_logs[job_id] = []
        log_message(job_id, f"[START] Converting: {title}")

        progress_tracker[job_id] = {
            'status': 'converting',
            'current': 0,
            'total': 0,
            'error': None
        }

        # Sanitize title
        safe_title = re.sub(r'[^\w\s-]', '', title)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        log_message(job_id, f"[INFO] Safe title: {safe_title}")

        # Create folder
        article_output_dir = os.path.join(OUTPUT_BASE_DIR, safe_title)

        if os.path.exists(article_output_dir):
            import shutil
            log_message(job_id, f"[INFO] Clearing existing folder: {article_output_dir}")
            shutil.rmtree(article_output_dir)

        os.makedirs(article_output_dir, exist_ok=True)
        log_message(job_id, f"[INFO] Created output folder: {article_output_dir}")

        # Clean content
        article = content.strip()
        article = re.sub(r'\s+', ' ', article)
        article = article.replace('"', '"').replace('"', '"')
        article = article.replace(''', "'").replace(''', "'")

        # Split into chunks
        chunks = split_text_into_chunks(article, hard_limit=hard_limit)
        progress_tracker[job_id]['total'] = len(chunks)
        log_message(job_id, f"[INFO] Split into {len(chunks)} chunks")

        # Generate audio
        chunk_files = []
        for i, chunk in enumerate(chunks):
            output_file = os.path.join(
                article_output_dir,
                f"{safe_title}_{i + 1:03d}.wav"
            )
            log_message(job_id, f"[CHUNK {i + 1}/{len(chunks)}] Generating audio...")
            generate_speech_from_api(chunk, voice="tara", output_file=output_file)
            chunk_files.append(output_file)
            progress_tracker[job_id]['current'] = i + 1
            log_message(job_id, f"[CHUNK {i + 1}/{len(chunks)}] ✓ Completed")

        # Combine
        log_message(job_id, f"[INFO] Combining {len(chunks)} chunks into MP3...")
        combine_audio_files(chunk_files, article_output_dir, safe_title)
        log_message(job_id, f"[SUCCESS] MP3 created: {safe_title}.mp3")

        progress_tracker[job_id]['status'] = 'completed'
        log_message(job_id, f"[END] Conversion completed successfully")

        # Move to completed
        with completed_jobs_lock:
            completed_jobs.append({
                'job_id': job_id,
                'title': title,
                'status': 'completed'
            })
            if len(completed_jobs) > 5:
                completed_jobs.pop(0)

    except Exception as e:
        progress_tracker[job_id]['error'] = str(e)
        log_message(job_id, f"[ERROR] {str(e)}")
        print(f"Error: {e}")

    finally:
        # Mark current job as done
        with current_job_lock:
            global current_job
            current_job = None


def combine_audio_files(wav_files, output_dir, title):
    """Combine multiple WAV files into a single MP3"""
    try:
        concat_file = os.path.join(output_dir, 'concat_list.txt')
        with open(concat_file, 'w') as f:
            for wav_file in wav_files:
                f.write(f"file '{wav_file}'\n")

        output_mp3 = os.path.join(output_dir, f"{title}.mp3")

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

        os.remove(concat_file)

    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
    except Exception as e:
        print(f"Error combining audio: {e}")


def queue_worker():
    """Background thread that processes queue"""
    while True:
        try:
            # Wait for a job (blocking)
            job = job_queue.get()

            with current_job_lock:
                global current_job
                current_job = job

            # Run the conversion
            convert_article_to_audio(job)

            job_queue.task_done()

        except Exception as e:
            print(f"Queue worker error: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/queue/status', methods=['GET'])
def queue_status():
    """Get current queue status"""
    with current_job_lock:
        curr = current_job.copy() if current_job else None

    queued = []
    for idx, job in enumerate(list(job_queue.queue), start=2):
        queued.append({
            'job_id': job['job_id'],
            'title': job['title'],
            'position': idx
        })

    completed = []
    with completed_jobs_lock:
        completed = [j.copy() for j in completed_jobs]

    return jsonify({
        'current': curr,
        'queued': queued,
        'completed': completed
    })


@app.route('/convert', methods=['POST'])
def convert():
    """Add job to queue"""
    data = request.json
    url = data.get('url', '').strip()
    hard_limit = int(data.get('hard_limit', 80))
    title = data.get('title', '')
    content = data.get('content', '')

    if not title or not content:
        if not url:
            return jsonify({'error': 'Either URL or title+content required'}), 400

        try:
            article, metadata = fetch_article(url)
            if not article:
                return jsonify({'error': 'Failed to extract article'}), 400
            title = metadata.title
            content = article
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Check queue size
    if job_queue.full():
        return jsonify({'error': 'Queue is full (max 10 jobs). Try again later.'}), 429

    # Create job
    job_id = get_next_job_id()
    job = {
        'job_id': job_id,
        'title': title,
        'content': content,
        'hard_limit': hard_limit,
        'status': 'queued'
    }

    try:
        job_queue.put_nowait(job)
        progress_tracker[job_id] = {
            'status': 'queued',
            'current': 0,
            'total': 0,
            'error': None
        }
        app_logs[job_id] = []

        return jsonify({
            'job_id': job_id,
            'position': job_queue.qsize() + (1 if current_job else 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/progress/<int:job_id>', methods=['GET'])
def get_progress(job_id):
    """Get progress for a specific job"""
    if job_id not in progress_tracker:
        return jsonify({'error': 'Invalid job ID'}), 404

    return jsonify(progress_tracker[job_id])


@app.route('/logs/<int:job_id>', methods=['GET'])
def get_logs(job_id):
    """Get logs for a specific job"""
    if job_id not in app_logs:
        return jsonify({'logs': []})

    return jsonify({'logs': app_logs[job_id]})


@app.route('/queue/cancel/<int:job_id>', methods=['POST'])
def cancel_job(job_id):
    """Cancel a queued job"""
    # Check if it's the current job
    with current_job_lock:
        if current_job and current_job['job_id'] == job_id:
            return jsonify({'error': 'Cannot cancel running job'}), 400

    # Try to remove from queue
    new_queue = []
    found = False
    for job in list(job_queue.queue):
        if job['job_id'] == job_id:
            found = True
        else:
            new_queue.append(job)

    if found:
        # Rebuild queue
        job_queue.queue.clear()
        for job in new_queue:
            job_queue.put_nowait(job)

        return jsonify({'success': True, 'message': 'Job cancelled'})
    else:
        return jsonify({'error': 'Job not found'}), 404


if __name__ == '__main__':
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Start queue worker thread
    worker_thread = threading.Thread(target=queue_worker, daemon=True)
    worker_thread.start()

    app.run(debug=False, host='0.0.0.0', port=5000)