# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
import os
import sys
import argparse
import logging
import tempfile
import shutil

logging.getLogger('matplotlib').setLevel(logging.WARNING)

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_data(model_output):
    try:
        for i in model_output:
            tts_speech = i['tts_speech']

            if hasattr(tts_speech, "detach"):
                tts_speech = tts_speech.detach()

            if hasattr(tts_speech, "is_cuda") and tts_speech.is_cuda:
                tts_speech = tts_speech.cpu()

            if hasattr(tts_speech, "dim") and tts_speech.dim() > 1:
                tts_speech = tts_speech.squeeze()

            tts_audio = (tts_speech.clamp(-1, 1).numpy() * (2 ** 15)).astype(np.int16).tobytes()
            yield tts_audio
    except Exception as e:
        import traceback
        print("[generate_data] ERROR:", e)
        traceback.print_exc()
        raise


def generate_data_with_cleanup(model_output, tmp_path: str):
    """
    在整个流式输出结束后，再删除临时文件
    """
    try:
        for chunk in generate_data(model_output):
            yield chunk
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                print(f"[cleanup] removed temp file: {tmp_path}")
        except Exception as e:
            print(f"[cleanup] failed to remove temp file {tmp_path}: {e}")


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output), media_type="application/octet-stream")


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(),
    prompt_text: str = Form(),
    prompt_wav: UploadFile = File()
):
    if "<|endofprompt|>" not in prompt_text:
        prompt_text = prompt_text + "<|endofprompt|>"

    suffix = os.path.splitext(prompt_wav.filename)[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        prompt_wav.file.seek(0)
        shutil.copyfileobj(prompt_wav.file, tmp)

    print(f"[zero_shot] tts_text={tts_text}")
    print(f"[zero_shot] prompt_text={prompt_text}")
    print(f"[zero_shot] temp prompt wav={tmp_path}")

    model_output = cosyvoice.inference_zero_shot(
        tts_text,
        prompt_text,
        tmp_path
    )

    return StreamingResponse(
        generate_data_with_cleanup(model_output, tmp_path),
        media_type="application/octet-stream"
    )


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output), media_type="application/octet-stream")


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output), media_type="application/octet-stream")


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output), media_type="application/octet-stream")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/media/ubuntu/data/pretrained_models/Fun-CosyVoice3-0.5B',
        help='local path or modelscope repo id'
    )
    args = parser.parse_args()

    cosyvoice = AutoModel(
        model_dir=args.model_dir,
        load_vllm=True
    )

    uvicorn.run(app, host="0.0.0.0", port=args.port)