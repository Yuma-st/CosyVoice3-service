# -*- coding: utf-8 -*-
"""
只测试 CosyVoice3 FastAPI SSE 服务的 zero-shot（服务里的 synthesis_type = preset）

输出：
- outputs/zero_shot.wav

用法示例：
python client_zero_shot.py --host 127.0.0.1 --port 50000
python client_zero_shot.py --host 127.0.0.1 --port 50000 --sr 24000
"""

import os
import json
import base64
import wave
import argparse
import requests
from typing import Dict, List


def save_pcm16_to_wav(pcm_bytes: bytes, wav_path: str, sample_rate: int = 24000):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)     # mono
        wf.setsampwidth(2)     # int16 => 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def collect_sse_pcm(url: str, payload: Dict, timeout: int = 300) -> bytes:
    """
    请求 SSE 接口，收集所有 PCM16 数据并拼接返回
    """
    print("=" * 80)
    print("POST", url)
    print("payload =", json.dumps(payload, ensure_ascii=False, indent=2))

    resp = requests.post(url, json=payload, stream=True, timeout=timeout)
    resp.raise_for_status()

    pcm_chunks: List[bytes] = []
    event_count = 0

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue

        line = raw_line.strip()
        if not line.startswith("data: "):
            continue

        event_count += 1
        data_str = line[6:]

        try:
            obj = json.loads(data_str)
        except Exception as e:
            print(f"[WARN] SSE JSON parse failed: {e}, line={line[:200]}")
            continue

        finish = bool(obj.get("finish", False))
        b64 = obj.get("data", "")

        if b64:
            try:
                pcm_chunks.append(base64.b64decode(b64))
            except Exception as e:
                print(f"[WARN] base64 decode failed: {e}")

        if finish:
            print(f"[INFO] finish event received after {event_count} events")
            break

    pcm_bytes = b"".join(pcm_chunks)
    print(f"[INFO] total pcm bytes = {len(pcm_bytes)}")
    return pcm_bytes


def run_zero_shot(base_url: str, output_wav: str, sample_rate: int,
                  tts_text: str, prompt_wav: str, prompt_text: str):
    """
    走你服务里的 zero-shot：synthesis_type = preset
    """
    payload = {
        "synthesis_type": "preset",
        "tts_text": tts_text,
    }

    # 只有传了才带上，方便走服务端默认 prompt
    if prompt_wav:
        payload["prompt_wav"] = prompt_wav
    if prompt_text:
        payload["prompt_text"] = prompt_text

    pcm = collect_sse_pcm(base_url, payload)
    save_pcm16_to_wav(pcm, output_wav, sample_rate)
    print(f"[OK] zero-shot wav saved to: {output_wav}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--sr", type=int, default=24000, help="output wav sample rate")
    parser.add_argument("--outdir", type=str, default="outputs")

    # 目标合成文本
    parser.add_argument(
        "--tts_text",
        type=str,
        default="今日我哋见到模型嘅本质，其实好多时候系将我哋人类嘅知识有效咁聚集埋一齐。能够成为今日我哋一个重要嘅智慧体，去帮助我哋嘅业务开发，帮我哋去应用行业嘅知识。今日我哋实际上系由一个资讯时代，真正踏入咗一个智能时代。喺智能时代入面，一个重要嘅环节就系模型变得无处不在，亦都系模型所代表嘅知识体系。",
        help="target text to synthesize"
    )

    # 可选：自定义 zero-shot 参考
    parser.add_argument(
        "--prompt_wav",
        type=str,
        default="/media/ubuntu/data/audio.wav",
        help="server-side prompt wav path; leave empty to use server default"
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="好的，我会提醒他尽快给您回电，并告知他事情的详细经过和您的具体解决方案。",
        help="transcript of prompt_wav; leave empty to use server default"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="zero_shot.wav",
        help="output wav filename"
    )

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1/tts"
    os.makedirs(args.outdir, exist_ok=True)
    output_wav = os.path.join(args.outdir, args.output)

    print(f"[INFO] server url: {base_url}")
    print(f"[INFO] output wav: {output_wav}")

    try:
        run_zero_shot(
            base_url=base_url,
            output_wav=output_wav,
            sample_rate=args.sr,
            tts_text=args.tts_text,
            prompt_wav=args.prompt_wav,
            prompt_text=args.prompt_text,
        )
    except Exception as e:
        print(f"[ERROR] zero-shot test failed: {e}")


if __name__ == "__main__":
    main()