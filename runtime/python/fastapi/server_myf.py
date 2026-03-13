# -*- coding: utf-8 -*-
# NVIDIA + vLLM CosyVoice3 FastAPI SSE Server
# Keep API aligned with your server_myf.py (/v1/tts, SSE, base64 PCM16)

import os
# vLLM 多进程建议 spawn（与你 NPU 版本一致的策略）
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 如果你想强制 vLLM V0（你日志里已经 fallback 到 V0 了）
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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{ROOT_DIR}/../../..")
sys.path.append(f"{ROOT_DIR}/../../../third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

# ========= 对齐你 NPU 服务的外部接口 =========
API_PATH = "/v1/tts"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 50000

# 你要求的固定 prompt（按你 NPU 服务一致）
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
    wav_float_1d: torch tensor 1D (float) in [-1, 1]
    输出：16-bit PCM bytes（小端）
    """
    if hasattr(wav_float_1d, "detach"):
        wav_float_1d = wav_float_1d.detach()

    # vLLM + CUDA tensor -> CPU
    if hasattr(wav_float_1d, "is_cuda") and wav_float_1d.is_cuda:
        wav_float_1d = wav_float_1d.cpu()

    if hasattr(wav_float_1d, "dim") and wav_float_1d.dim() > 1:
        wav_float_1d = wav_float_1d.squeeze()

    wav_float_1d = torch.clamp(wav_float_1d, -1.0, 1.0)
    pcm16 = (wav_float_1d * (2 ** 15)).to(dtype=torch.int16)
    return pcm16.numpy().tobytes()


def _get_prompt_text_with_eop(text: str) -> str:
    text = text or ""
    if "<|endofprompt|>" not in text:
        text = text + "<|endofprompt|>"
    return text


def _stream_infer_sse(req: Dict[str, Any]):
    """
    生成 SSE：每个 chunk 输出 base64 PCM16
    """
    if cosyvoice is None:
        raise RuntimeError("model_not_loaded")

    synthesis_type = (req.get("synthesis_type") or "").strip()
    tts_text = req.get("tts_text") or ""
    prompt_index = req.get("prompt_index")
    instruct_text = req.get("instruct_text")

    # ===== 路由选择（对齐你的 NPU 服务）=====
    if synthesis_type == "preset":
        prompt_text = _get_prompt_text_with_eop(DEFAULT_PROMPT_TEXT)

        # 兼容两种用法：如果 AutoModel 支持传路径就传路径；否则读成 16k numpy
        try:
            model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, DEFAULT_PROMPT_WAV)
        except Exception:
            prompt_speech_16k = load_wav(DEFAULT_PROMPT_WAV, 16000)
            model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)

    elif synthesis_type in ("instruct", "dialect", "cross_lingual"):
        if not prompt_index:
            raise ValueError("prompt_index is required for instruct/dialect/cross_lingual")
        if not instruct_text:
            raise ValueError("instruct_text is required for instruct/dialect/cross_lingual")
        model_output = cosyvoice.inference_instruct(tts_text, prompt_index, instruct_text)

    else:
        raise ValueError(f"unsupported synthesis_type: {synthesis_type}")

    # ===== SSE 流式输出 =====
    try:
        for out in model_output:
            pcm16_bytes = _float_to_pcm16_bytes(out["tts_speech"])
            b64 = base64.b64encode(pcm16_bytes).decode("ascii")
            yield _sse_pack(b64, finish=False)
    finally:
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

    # 对齐你 NPU 服务：这三类要求 prompt_index
    if synthesis_type in ("instruct", "dialect", "cross_lingual") and not req.get("prompt_index"):
        raise HTTPException(status_code=422, detail="missing prompt_index")

    headers = {
        "X-qsize": "0",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        _stream_infer_sse(req),
        media_type="text/event-stream",
        headers=headers,
    )


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
    parser.add_argument("--load_trt", action="store_true", default=True)  # ✅ 默认关掉，避免 tensorrt 依赖
    parser.add_argument("--fp16", action="store_true", default=False)
    args = parser.parse_args()

    print("Torch CUDA available:", torch.cuda.is_available())
    # ✅ 这里按你的目标初始化：vLLM + (可选 TRT)
    cosyvoice = AutoModel(
        model_dir=args.model_dir,
        load_trt=args.load_trt,
        load_vllm=args.load_vllm,
        fp16=args.fp16,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

