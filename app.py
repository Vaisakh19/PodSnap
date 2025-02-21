from fastapi import FastAPI, File, UploadFile, HTTPException
import whisper
from transformers import pipeline
import os
import re
import logging
import time

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load AI models
whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}

# Max file size (in bytes) → 200MB (approx. 2-hour podcast)
MAX_FILE_SIZE = 200 * 1024 * 1024

def sanitize_filename(filename: str) -> str:
    """Removes special characters from filename to prevent issues."""
    return re.sub(r'[^\w.-]', '_', filename)

@app.get("/")
def read_root():
    return {"message": "API is running!"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    start_time = time.time()  # Start timing

    # Validate file extension
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        logging.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file type. Only MP3, WAV, and M4A are allowed.")

    # Check file size before saving
    file_size = 0
    content = await file.read()
    file_size = len(content)

    if file_size > MAX_FILE_SIZE:
        logging.warning(f"File too large: {file.filename} ({file_size / (1024 * 1024):.2f} MB)")
        raise HTTPException(status_code=400, detail="File is too large. Maximum allowed size is 200MB.")

    # Save file securely
    safe_filename = sanitize_filename(file.filename)
    file_path = os.path.join("temp", safe_filename)

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        logging.info(f"File received: {file.filename} ({file_size / (1024 * 1024):.2f} MB)")

        # Perform transcription
        transcription = whisper_model.transcribe(file_path)["text"]

        # Summarization
        transcription_length = len(transcription.split())
        max_length = max(15, int(transcription_length * 0.5))
        min_length = max(5, int(transcription_length * 0.2))
        min_length = min(min_length, max_length - 5)

        summary = summarizer(transcription, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]

        logging.info(f"Transcription completed for {file.filename} in {time.time() - start_time:.2f} seconds")

        return {"transcription": transcription, "summary": summary}

    except Exception as e:
        logging.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during transcription.")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)  # Clean up uploaded file
