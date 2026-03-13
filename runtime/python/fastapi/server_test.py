# -*- coding: utf-8 -*-
# NVIDIA + vLLM CosyVoice3 FastAPI SSE Server
# SSE output: base64 PCM16
# API:
#   POST /v1/tts
#   GET  /health
#
# 支持：
# 1. preset        -> zero-shot，可在请求中自定义 prompt_wav / prompt_text
# 2. instruct      -> 预训练音色 + instruct
# 3. dialect       -> 预训练音色 + instruct
# 4. cross_lingual -> 预训练音色 + instruct
#
# 运行：
#   python server_test.py
#
# 示例请求（zero-shot，自定义粤语参考）：
# curl -N -X POST "http://127.0.0.1:50000/v1/tts" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "synthesis_type": "preset",
#     "tts_text": "今日天气几好，我哋一阵去饮茶啦。",
#     "prompt_wav": "/media/ubuntu/data/CosyVoice/asset/cantonese_prompt.wav",
#     "prompt_text": "各方面受到一啲行业影响会比较大，比如教育，即系补习社。"
#   }'

import os

# vLLM 多进程建议 spawn
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 如需强制 vLLM V0，可打开下面这一行
# os.environ["VLLM_USE_V1"] = "0"

import sys
import time
import json
import base64
import socket
import argparse
import logging
from typing import Optional, Any, Dict

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{ROOT_DIR}/../../..")
sys.path.append(f"{ROOT_DIR}/../../../third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

API_PATH = "/v1/tts"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 50000

# 默认 zero-shot prompt（当请求里不传 prompt_wav / prompt_text 时兜底）
DEFAULT_PROMPT_WAV = "/media/ubuntu/data/CosyVoice/asset/zero_shot_prompt.wav"
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cosyvoice: Optional[Any] = None
_device_name = socket.gethostname()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sse_pack(data_b64: str, finish: bool) -> str:
    payload = {
        "data": data_b64,
        "timestamp": _now_ms(),
        "finish": bool(finish),
        "device": _device_name,
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _float_to_pcm16_bytes(wav_float_1d: Any) -> bytes:
    """
    wav_float_1d: torch tensor 1D / 2D (float) in [-1, 1]
    输出：16-bit PCM bytes（小端）
    """
    if hasattr(wav_float_1d, "detach"):
        wav_float_1d = wav_float_1d.detach()

    if hasattr(wav_float_1d, "is_cuda") and wav_float_1d.is_cuda:
        wav_float_1d = wav_float_1d.cpu()

    if hasattr(wav_float_1d, "dim") and wav_float_1d.dim() > 1:
        wav_float_1d = wav_float_1d.squeeze()

    wav_float_1d = torch.clamp(wav_float_1d, -1.0, 1.0)
    pcm16 = (wav_float_1d * (2 ** 15)).to(dtype=torch.int16)
    return pcm16.numpy().tobytes()


def _get_prompt_text_with_eop(text: str) -> str:
    text = (text or "").strip()
    if "<|endofprompt|>" not in text:
        text = text + "<|endofprompt|>"
    return text


def _resolve_prompt_text(req: Dict[str, Any]) -> str:
    prompt_text = (req.get("prompt_text") or "").strip()
    if not prompt_text:
        prompt_text = DEFAULT_PROMPT_TEXT
    return _get_prompt_text_with_eop(prompt_text)


def _resolve_prompt_wav(req: Dict[str, Any]) -> str:
    prompt_wav = (req.get("prompt_wav") or "").strip()
    if not prompt_wav:
        prompt_wav = DEFAULT_PROMPT_WAV

    if not os.path.isfile(prompt_wav):
        raise ValueError(f"prompt_wav not found: {prompt_wav}")

    return prompt_wav


def _stream_infer_sse(req: Dict[str, Any]):
    """
    生成 SSE：每个 chunk 输出 base64 PCM16
    """
    if cosyvoice is None:
        raise RuntimeError("model_not_loaded")

    synthesis_type = (req.get("synthesis_type") or "").strip()
    tts_text = (req.get("tts_text") or "").strip()
    prompt_index = req.get("prompt_index")
    instruct_text = req.get("instruct_text")

    logging.info(
        "req_begin synthesis_type=%s tts_text=%s",
        synthesis_type,
        tts_text[:120]
    )

    if synthesis_type == "preset":
        prompt_wav = _resolve_prompt_wav(req)
        prompt_text = _resolve_prompt_text(req)

        logging.info(
            "zero_shot prompt_wav=%s prompt_text=%s",
            prompt_wav,
            prompt_text[:120]
        )

        # 兼容两种用法：
        # 1) 某些 AutoModel 版本支持直接传路径
        # 2) 不支持时，回退为 load_wav(..., 16000)
        try:
            model_output = cosyvoice.inference_zero_shot(
                tts_text,
                prompt_text,
                prompt_wav
            )
        except Exception as e:
            logging.warning("inference_zero_shot(path) failed, fallback to load_wav: %s", e)
            prompt_speech_16k = load_wav(prompt_wav, 16000)
            model_output = cosyvoice.inference_zero_shot(
                tts_text,
                prompt_text,
                prompt_speech_16k
            )

    elif synthesis_type in ("instruct", "dialect", "cross_lingual"):
        if not prompt_index:
            raise ValueError("prompt_index is required for instruct/dialect/cross_lingual")
        if not instruct_text:
            raise ValueError("instruct_text is required for instruct/dialect/cross_lingual")

        logging.info(
            "instruct prompt_index=%s instruct_text=%s",
            str(prompt_index),
            str(instruct_text)[:120]
        )

        model_output = cosyvoice.inference_instruct(
            tts_text,
            prompt_index,
            instruct_text
        )

    else:
        raise ValueError(f"unsupported synthesis_type: {synthesis_type}")

    chunk_count = 0
    try:
        for out in model_output:
            if "tts_speech" not in out:
                continue
            pcm16_bytes = _float_to_pcm16_bytes(out["tts_speech"])
            b64 = base64.b64encode(pcm16_bytes).decode("ascii")
            chunk_count += 1
            yield _sse_pack(b64, finish=False)
    finally:
        logging.info("req_end synthesis_type=%s chunks=%d", synthesis_type, chunk_count)
        yield _sse_pack("", finish=True)


@app.get("/health")
async def health():
    if cosyvoice is None:
        return JSONResponse({"ok": False, "reason": "model_not_loaded"}, status_code=503)
    cuda_ok = torch.cuda.is_available()
    return {"ok": True, "device": _device_name, "cuda": cuda_ok}


@app.post(API_PATH)
async def unified_tts(request: Request):
    try:
        req = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_json")

    synthesis_type = (req.get("synthesis_type") or "").strip()
    tts_text = req.get("tts_text")

    if not synthesis_type:
        raise HTTPException(status_code=422, detail="missing synthesis_type")

    if not isinstance(tts_text, str) or not tts_text.strip():
        raise HTTPException(status_code=422, detail="missing tts_text")

    if synthesis_type == "preset":
        prompt_wav = (req.get("prompt_wav") or "").strip()
        prompt_text = (req.get("prompt_text") or "").strip()

        # 允许都不传：走默认 DEFAULT_PROMPT_WAV / DEFAULT_PROMPT_TEXT
        # 但如果传了自定义 prompt_wav，则要求文件存在，且建议 prompt_text 一并传入
        if prompt_wav and (not os.path.isfile(prompt_wav)):
            raise HTTPException(status_code=422, detail=f"prompt_wav not found: {prompt_wav}")

        if prompt_wav and not prompt_text:
            raise HTTPException(status_code=422, detail="missing prompt_text for custom prompt_wav")

    elif synthesis_type in ("instruct", "dialect", "cross_lingual"):
        if not req.get("prompt_index"):
            raise HTTPException(status_code=422, detail="missing prompt_index")
        if not req.get("instruct_text"):
            raise HTTPException(status_code=422, detail="missing instruct_text")

    else:
        raise HTTPException(status_code=422, detail=f"unsupported synthesis_type: {synthesis_type}")

    headers = {
        "X-qsize": "0",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }

    try:
        return StreamingResponse(
            _stream_infer_sse(req),
            media_type="text/event-stream",
            headers=headers,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logging.exception("unified_tts failed")
        raise HTTPException(status_code=500, detail=f"internal_error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/media/ubuntu/data/pretrained_models/Fun-CosyVoice3-0.5B",
        help="local path or modelscope repo id",
    )
    parser.add_argument("--load_vllm", action="store_true", default=True)
    parser.add_argument("--load_trt", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    args = parser.parse_args()

    print("Torch CUDA available:", torch.cuda.is_available())

    cosyvoice = AutoModel(
        model_dir=args.model_dir,
        load_trt=args.load_trt,
        load_vllm=args.load_vllm,
        fp16=args.fp16,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")