"""CLI: run a self-hosted voice session for one avatar.

    python orchestrator_main.py --avatar 4 --channel my-test [--idle 180] [--max 600]

Reads Agora + Deepgram + Cartesia keys from backend/.env, avatar voice from
backend/avatars.json, joins the channel as uid 777 and runs the full loop.
"""
import argparse
import json
import re
import sys

import requests

from session import VoiceSession

BACKEND = "/root/AvatarGarden/backend"
BASE = "https://avatars.sympoiesis.xyz"


def load_env():
    env = {}
    for line in open(f"{BACKEND}/.env"):
        m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2)
    return env


ORCH_UID = 777  # the token must be minted for the SAME uid the SDK connects with


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", required=True)
    ap.add_argument("--channel", required=True)
    ap.add_argument("--idle", type=int, default=180)
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--codec", default="g722", choices=["g722", "opus"])
    ap.add_argument("--rtp-port", type=int, default=0, help="UDP listen port (rtp transport)")
    ap.add_argument("--ws-port", type=int, default=0, help="WS listen port (ws transport)")
    ap.add_argument("--full-duplex", action="store_true",
                    help="web client (browser AEC): keep uplink during downlink, allow barge-in")
    ap.add_argument("--no-tts-streaming", action="store_true",
                    help="disable Cartesia SSE streaming (fetch each full sentence)")
    args = ap.parse_args()

    env = load_env()
    avatars = json.load(open(f"{BACKEND}/avatars.json"))
    av = next((a for a in avatars if str(a["id"]) == str(args.avatar)), None)
    if not av:
        print(f"avatar {args.avatar} not found"); return 1

    creds = requests.get(
        f"{BASE}/api/voice/token?channel={args.channel}&uid={ORCH_UID}").json()
    print(f"[main] token for {args.channel} uid {ORCH_UID}", flush=True)

    _ld = av.get("llmDefaults", {})

    s = VoiceSession(
        avatar_id=args.avatar, channel=args.channel, token=creds["token"],
        app_id=creds["appId"], deepgram_key=env["DEEPGRAM_API_KEY"],
        cartesia_key=env["CARTESIA_API_KEY"],
        tts_voice_id=av.get("ttsVoiceId") or "default",
        tts_streaming=(not args.no_tts_streaming) and bool(
            _ld.get("streaming", _ld.get("ttsStreaming", True))),
        full_duplex=args.full_duplex, codec=args.codec,
        transport=("ws" if args.ws_port else ("rtp" if args.rtp_port else "agora")),
        rtp_port=args.rtp_port, ws_port=args.ws_port)
    s.run(idle_timeout_s=args.idle, max_duration_s=args.max)
    return 0


if __name__ == "__main__":
    sys.exit(main())