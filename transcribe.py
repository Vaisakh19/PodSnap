import whisper
from transformers import pipeline
model = whisper.load_model("base")
result = model.transcribe("K.  lyrics edit  #cas #audio #lyrics #overlayedit #explore #foryou #edit.mp3")
transcript=result["text"]
transcription_length=len(transcript.split())

if transcription_length < 50:  
    max_factor, min_factor = 0.8, 0.4  

elif transcription_length < 300:  
    max_factor, min_factor = 0.5, 0.2  

else:  
    max_factor, min_factor = 0.3, 0.1  

max_length = max(15, int(transcription_length * max_factor))
min_length = max(5, int(transcription_length * min_factor))

min_length = min(min_length, max_length - 5)
with open('transcription.txt', 'w',encoding="utf-8") as f:
    f.write(transcript)
summarizer=pipeline("summarization",model="facebook/bart-large-cnn")
summary=summarizer(transcript,max_length=150,min_length=50,do_sample=False)[0]["summary_text"]
with open('summary.txt', 'w',encoding="utf-8") as f:
    f.write(summary)
    
print("Transcription saved successfully to transcription.txt")
print("Summary saved successfully to summary.txt")
