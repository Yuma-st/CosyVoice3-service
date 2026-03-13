#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loadtest_sse_tts.py
- Target: FastAPI SSE endpoint: POST /v1/tts (text/event-stream)
- Parse SSE lines: 'data: {json}\n\n'
- JSON payload: {"data": "<base64 pcm16>", "finish": bool, "timestamp": ms, "device": "..."}
- Collect:
  - TTFF (time to first SSE data event)
  - Total latency
  - Total PCM bytes (decoded from base64 chunks)
- Optional: save WAV per request (PCM16 mono) if --save-wav is set.
"""

import argparse
import asyncio
import base64
import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import httpx
import numpy as np


@dataclass
class OneResult:
    ok: bool
    status_code: int
    err: str
    ttff_ms: float
    total_ms: float
    pcm_bytes: int


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    # nearest-rank-ish, linear interpolation
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _write_wav(path: str, pcm16_bytes: bytes, sample_rate: int = 22050) -> None:
    """
    CosyVoice server 输出的是 PCM16 bytes（单声道），采样率 server 没显式返回。
    你如果知道模型实际采样率（例如 22050/24000/16000），改这里即可。
    """
    import wave

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16_bytes)


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    timeout_s: float,
    save_wav: bool,
    wav_dir: str,
    wav_sr: int,
    req_id: int,
) -> OneResult:
    t0 = time.perf_counter()
    first_event_t: Optional[float] = None
    pcm_buf = bytearray()

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
    }

    try:
        # stream=True: use client.stream context
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=timeout_s,
        ) as resp:
            status = resp.status_code
            if status != 200:
                body = (await resp.aread())[:512]
                return OneResult(False, status, f"http_{status}: {body!r}", 0.0, (time.perf_counter() - t0) * 1000, 0)

            # SSE is line-based; httpx provides aiter_lines()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue

                now = time.perf_counter()
                if first_event_t is None:
                    first_event_t = now

                # strip "data:" prefix
                s = line[len("data:"):].strip()
                try:
                    obj = json.loads(s)
                except Exception as e:
                    return OneResult(False, status, f"bad_json: {e}", 0.0, (time.perf_counter() - t0) * 1000, len(pcm_buf))

                b64 = obj.get("data", "")
                finish = bool(obj.get("finish", False))

                if b64:
                    try:
                        pcm_chunk = base64.b64decode(b64)
                        pcm_buf.extend(pcm_chunk)
                    except Exception as e:
                        return OneResult(False, status, f"bad_base64: {e}", 0.0, (time.perf_counter() - t0) * 1000, len(pcm_buf))

                if finish:
                    break

        t1 = time.perf_counter()
        ttff_ms = ((first_event_t - t0) * 1000) if first_event_t else 0.0
        total_ms = (t1 - t0) * 1000

        if save_wav:
            out = os.path.join(wav_dir, f"req_{req_id:05d}.wav")
            _write_wav(out, bytes(pcm_buf), sample_rate=wav_sr)

        ok = len(pcm_buf) > 0  # 没音频也算失败（便于你发现服务异常）
        err = "" if ok else "empty_audio"
        return OneResult(ok, 200, err, ttff_ms, total_ms, len(pcm_buf))

    except Exception as e:
        return OneResult(False, 0, f"exception: {type(e).__name__}: {e}", 0.0, (time.perf_counter() - t0) * 1000, len(pcm_buf))


async def main_async(args: argparse.Namespace) -> int:
    url = args.url.rstrip("/")
    # default endpoint
    if url.endswith("/v1/tts") is False:
        url = url + "/v1/tts"

    # request payload (preset path uses DEFAULT_PROMPT_TEXT/WAV in server)
    payload = {
        "synthesis_type": args.synthesis_type,
        "tts_text": args.text,
    }
    # for instruct/dialect/cross_lingual, server requires prompt_index + instruct_text
    if args.synthesis_type in ("instruct", "dialect", "cross_lingual"):
        payload["prompt_index"] = args.prompt_index
        payload["instruct_text"] = args.instruct_text

    limits = httpx.Limits(
        max_connections=max(args.concurrency * 2, 50),
        max_keepalive_connections=max(args.concurrency, 20),
    )

    # 注意：timeout 是“总超时”，SSE 很长时建议加大
    timeout = httpx.Timeout(timeout=args.timeout, connect=min(10.0, args.timeout))
    results: List[OneResult] = []

    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=False) as client:

        async def runner(i: int):
            async with sem:
                r = await _one_request(
                    client=client,
                    url=url,
                    payload=payload,
                    timeout_s=args.timeout,
                    save_wav=args.save_wav,
                    wav_dir=args.wav_dir,
                    wav_sr=args.wav_sr,
                    req_id=i,
                )
                results.append(r)

        t_start = time.perf_counter()
        tasks = [asyncio.create_task(runner(i)) for i in range(args.total)]
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    # summarize
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    wall_s = t_end - t_start
    qps = (len(ok) / wall_s) if wall_s > 0 else 0.0

    lat = sorted([r.total_ms for r in ok])
    ttff = sorted([r.ttff_ms for r in ok])
    pcm = [r.pcm_bytes for r in ok]

    def stat_block(name: str, xs: List[float]) -> str:
        if not xs:
            return f"{name}: n=0"
        return (
            f"{name}: n={len(xs)} "
            f"avg={statistics.mean(xs):.1f}ms "
            f"p50={_percentile(xs,50):.1f}ms "
            f"p90={_percentile(xs,90):.1f}ms "
            f"p95={_percentile(xs,95):.1f}ms "
            f"p99={_percentile(xs,99):.1f}ms "
            f"max={max(xs):.1f}ms"
        )

    print("=" * 80)
    print(f"URL          : {url}")
    print(f"mode         : {args.synthesis_type}")
    print(f"concurrency  : {args.concurrency}")
    print(f"total        : {args.total}")
    print(f"wall_time    : {wall_s:.2f}s")
    print(f"success      : {len(ok)}")
    print(f"failed       : {len(bad)}")
    print(f"QPS(ok)      : {qps:.3f}/s")
    if pcm:
        print(f"audio_bytes  : avg={statistics.mean(pcm):.0f} min={min(pcm)} max={max(pcm)} sum={sum(pcm)}")
    print(stat_block("TTFF", ttff))
    print(stat_block("LAT ", lat))

    if bad:
        # show top few errors
        from collections import Counter
        c = Counter([b.err for b in bad])
        print("-" * 80)
        print("Top errors:")
        for k, v in c.most_common(10):
            print(f"  {v:4d}  {k}")

    print("=" * 80)

    return 0 if len(bad) == 0 else 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concurrent load test for SSE TTS /v1/tts")
    p.add_argument("--url", type=str, default="http://127.0.0.1:50000", help="base url, e.g. http://127.0.0.1:50000")
    p.add_argument("-c", "--concurrency", type=int, default=4, help="concurrent in-flight requests")
    p.add_argument("-n", "--total", type=int, default=20, help="total requests")
    p.add_argument("--timeout", type=float, default=120.0, help="per-request timeout seconds (SSE can be long)")
    p.add_argument("--synthesis_type", type=str, default="preset",
                   choices=["preset", "instruct", "dialect", "cross_lingual"],
                   help="match server_myf.py routing")
    p.add_argument("--text", type=str, default="你好，我是CosyVoice3。", help="tts_text")
    # only for instruct-like
    p.add_argument("--prompt_index", type=str, default="圆圆", help="required for instruct/dialect/cross_lingual")
    p.add_argument("--instruct_text", type=str, default='{"emotion":"平和","speed":"正常"}', help="required for instruct/dialect/cross_lingual")
    # optional wav save
    p.add_argument("--save-wav", action="store_true", help="save wav per request (will slow down load test)")
    p.add_argument("--wav-dir", type=str, default="out_wavs", help="output dir if --save-wav")
    p.add_argument("--wav-sr", type=int, default=22050, help="wav sample rate used when saving")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(asyncio.run(main_async(args)))
