# -*- coding: utf-8 -*-
"""
测试原始 server.py 的 /inference_zero_shot 接口
返回的是原始 PCM16 字节流，这里会自动保存成可试听 wav 文件。

用法示例：
python client_zero_shot_raw.py \
  --host 127.0.0.1 \
  --port 50000 \
  --tts_text "你好啊，今日过得点呀？食咗饭未？" \
  --prompt_text "各方面受到一啲行业影响会比较大，比如教育，即系补习社。" \
  --prompt_wav "/media/ubuntu/data/CosyVoice/asset/cantonese_prompt.wav"

如果不传参数，会用脚本里的默认值。
"""

import os
import wave
import argparse
import requests


def save_pcm16_to_wav(pcm_bytes: bytes, wav_path: str, sample_rate: int = 24000):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)      # 单声道
        wf.setsampwidth(2)      # int16 -> 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--tts_text", type=str,
                        default="你好啊，今日过得点呀？食咗饭未？")
    parser.add_argument("--prompt_text", type=str,
                        default="各方面受到一啲行业影响会比较大，比如教育，即系补习社。")
    parser.add_argument("--prompt_wav", type=str,
                        default="/media/ubuntu/data/CosyVoice 3.0-1.5B.wav")
    parser.add_argument("--sr", type=int, default=24000,
                        help="输出 wav 采样率，CosyVoice 常见为 24000")
    parser.add_argument("--output", type=str, default="outputs/zero_shot_raw.wav")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/inference_zero_shot"

    if not os.path.isfile(args.prompt_wav):
        raise FileNotFoundError(f"prompt_wav not found: {args.prompt_wav}")

    print("[INFO] url       :", url)
    print("[INFO] tts_text  :", args.tts_text)
    print("[INFO] prompt_txt:", args.prompt_text)
    print("[INFO] prompt_wav:", args.prompt_wav)
    print("[INFO] output    :", args.output)

    with open(args.prompt_wav, "rb") as f:
        files = {
            "prompt_wav": (os.path.basename(args.prompt_wav), f, "audio/wav")
        }
        data = {
            "tts_text": args.tts_text,
            "prompt_text": args.prompt_text,
        }

        resp = requests.post(url, data=data, files=files, stream=True, timeout=300)
        print("[INFO] status_code:", resp.status_code)
        resp.raise_for_status()

        pcm_chunks = []
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                pcm_chunks.append(chunk)

        pcm_bytes = b"".join(pcm_chunks)
        print("[INFO] received bytes:", len(pcm_bytes))

    if len(pcm_bytes) == 0:
        raise RuntimeError("received empty audio bytes")

    save_pcm16_to_wav(pcm_bytes, args.output, args.sr)
    print("[OK] saved wav to:", args.output)


if __name__ == "__main__":
    main()