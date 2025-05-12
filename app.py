
import io
import json
import tempfile
import os
import torch
import torchaudio
import warnings
from pydub import AudioSegment
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from seamless_communication.inference import Translator
import uvicorn

# Silence warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Translation API")

# Initialize Translator
model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"

translator = Translator(
    model_name,
    vocoder_name,
    device=torch.device("cuda:0"),
    dtype=torch.float16,
)

@app.get("/")
def root():
    return {"message": "Welcome to the Translation API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/s2tt")
async def speech_to_text_translation(
    audio: UploadFile = File(...),
    src_lang: str = Form(...),
    tgt_lang: str = Form(...)
):
    try:
        if not audio.filename.endswith(".wav"):
            # Convert to wav using pydub
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_data = await audio.read()
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            audio_segment.export(temp.name, format="wav")
            input_path = temp.name
        else:
            input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with open(input_path, "wb") as f:
                f.write(await audio.read())

        # Resample to 16kHz if necessary
        waveform, orig_freq = torchaudio.load(input_path)
        if orig_freq != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq, 16000)
            torchaudio.save(input_path, waveform, 16000)

        # Translate: Calling the translator's prediction method
        text_output, _ = translator.predict(
            input=input_path,
            task_str="s2tt",
            tgt_lang=tgt_lang
        )
        os.remove(input_path)  # Clean up the temporary file

        # Ensure we only return a string as the response
        return {"translated_text": str(text_output[0])}

    except Exception as e:
        os.remove(input_path) if os.path.exists(input_path) else None
        raise HTTPException(status_code=500, detail=f"Error in S2TT: {str(e)}")

class T2TTRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

@app.post("/t2tt")
def text_to_text_translation(req: T2TTRequest):
    try:
        # Translate: Calling the translator's prediction method
        text_output, _ = translator.predict(
            input=req.text,
            task_str="t2tt",
            tgt_lang=req.tgt_lang,
            src_lang=req.src_lang
        )

        # Ensure we return a string as the response
        return {"translated_text": str(text_output[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in T2TT: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)