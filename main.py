import hashlib
import subprocess
import time
import wave

import pyaudio
import whisper

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
subprocess.call(['python.exe', 'audio_capture.py'], shell=True)
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")
frames = []
seconds = 5
for i in range(0, int(RATE / CHUNK * seconds)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")
stream.stop_stream()
stream.close()
p.terminate()

unique_number = int(hashlib.sha1(str(time.time()).encode('utf-8')).hexdigest(), 16) % (10 ** 8)

audio_file_name = f"output-{unique_number}.wav"

wf = wave.open(f"audio_outputs/{audio_file_name}", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

model = whisper.load_model("small")

result = model.transcribe(f'audio_outputs/{audio_file_name}', fp16=False, language="en", verbose=True)
print(result['text'])

# [00:00.000 --> 00:04.000]  The stale smell of old-beer lingers.
# [00:04.000 --> 00:06.000]  It takes heat to bring out the odor.
# [00:06.000 --> 00:09.000]  A cold dip restores health and zest.
# [00:09.000 --> 00:12.000]  A salt pickle tastes fine with ham.
# [00:12.000 --> 00:14.000]  Tacos al pastor are my favorite.
# [00:14.000 --> 00:18.000]  A zestful food is the hot cross bun.
#  The stale smell of old-beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest.
#  A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun.

for i, seg in enumerate(result['segments']):
    print(i + 1, "- ", seg['text'])

# 1 -   The stale smell of old-beer lingers.
# 2 -   It takes heat to bring out the odor.
# 3 -   A cold dip restores health and zest.
# 4 -   A salt pickle tastes fine with ham.
# 5 -   Tacos al pastor are my favorite.
# 6 -   A zestful food is the hot cross bun.

# language detection

audio = whisper.load_audio(f"audio_outputs/{audio_file_name}")
audio = whisper.pad_or_trim(audio)

mel = whisper.log_mel_spectrogram(audio).to(model.device)

_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("You are an advanced grammar evaluator designed to assess the grammatical and"
                                      "accuracy of written text. Generate only a score out of 5 for the candidate "
                                      "response, "
                                      "candidate response: {candidate_response} against the question, "
                                      "question: {question}")

llm = ChatOpenAI(openai_api_key="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", model_name="gpt-3.5-turbo")
candidate_response = result['text']
question = "The stale smell of old-beer lingers."

text = prompt.format(candidate_response=candidate_response, question=question)

print(llm.invoke(text, max_tokens=100))
