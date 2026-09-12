"""
Fase 3+4 E2E test: a bot plays the user; the orchestrator runs the full loop.

user bot (uid 2): publishes a Cartesia-TTS question in real time + streamed
silence (endpointing needs *received* silence), captures the orchestrator's
downlink frames (decoded PCM via its audio observer), measures latency, and
batch-ASRs the captured audio via Deepgram to verify the reply content.

Run after starting the orchestrator on the same channel:
    python orchestrator_main.py --avatar 4 --channel <ch> --uid 1
    /root/agora-orchestrator-venv/bin/python test_orchestrator_e2e.py --channel <ch>
"""
import argparse
import json
import re
import sys
import time
import wave

import requests

from agora.rtc.agora_service import AgoraService, AgoraServiceConfig
from agora.rtc.rtc_connection import RTCConnConfig, RtcConnectionPublishConfig
from agora.rtc.rtc_connection_observer import IRTCConnectionObserver
from agora.rtc.agora_base import (
    AudioProfileType, AudioScenarioType, AudioPublishType, VideoPublishType,
    AudioSubscriptionOptions, ClientRoleType, ChannelProfileType,
)
from agora.rtc.audio_frame_observer import IAudioFrameObserver

RATE = 16000
BASE = "https://avatars.sympoiesis.xyz"
BACKEND = "/root/AvatarGarden/backend"


def load_env():
    env = {}
    for line in open(f"{BACKEND}/.env"):
        m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True)
    ap.add_argument("--uid", default=2)
    ap.add_argument("--avatar", default="4")
    args = ap.parse_args()

    env = load_env()
    avatars = json.load(open(f"{BACKEND}/avatars.json"))
    av = next(a for a in avatars if str(a["id"]) == str(args.avatar))
    creds = requests.get(f"{BASE}/api/voice/token?channel={args.channel}&uid={args.uid}").json()

    # question synthesized with the avatar's own Cartesia voice
    q = requests.post("https://api.cartesia.ai/tts/bytes",
        headers={"X-API-Key": env["CARTESIA_API_KEY"], "Content-Type": "application/json",
                 "Cartesia-Version": "2026-08-14"},
        json={"transcript": "Voce pode me contar uma historia curta sobre o rio Marumbi?",
              "model_id": "sonic-3",
              "voice": av.get("ttsVoiceId") or "9904416a-0831-44ea-b8ee-5f145e8f9bbf",
              "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": RATE}},
        timeout=30)
    q.raise_for_status()
    question_pcm = q.content
    print(f"[bot] question: {len(question_pcm)} B PCM ({len(question_pcm) / 32000:.1f}s)",
          flush=True)

    captured = []
    marks = {"t_start": None, "downlink_started": None}

    class Rec(IAudioFrameObserver):
        def on_playback_audio_frame_before_mixing(self, _l, _c, uid, frame, _v=0, _d=None):
            if int(uid) == 777:
                now = time.monotonic()
                if marks["downlink_started"] is None:
                    marks["downlink_started"] = now
                    latency = now - marks["t_start"]
                    print(f"[bot] first downlink frame +{latency:.2f}s after question start",
                          flush=True)
                captured.append(bytes(frame.buffer))
            return 1

    class Obs(IRTCConnectionObserver):
        def on_connected(self, *_a, **_k):
            print("[bot] connected", flush=True)

        def on_user_joined(self, conn, uid):
            print(f"[bot] user joined: {uid}", flush=True)

    svc_cfg = AgoraServiceConfig()
    svc_cfg.app_id = creds["appId"]
    svc = AgoraService()
    assert svc.initialize(svc_cfg) == 0
    conn_cfg = RTCConnConfig(
        client_role_type=ClientRoleType.CLIENT_ROLE_BROADCASTER,
        channel_profile=ChannelProfileType.CHANNEL_PROFILE_LIVE_BROADCASTING,
        auto_subscribe_audio=1, auto_subscribe_video=0, audio_recv_media_packet=0,
        audio_subs_options=AudioSubscriptionOptions(
            packet_only=0, pcm_data_only=1, bytes_per_sample=2,
            number_of_channels=1, sample_rate_hz=RATE))
    pub_cfg = RtcConnectionPublishConfig(
        audio_profile=AudioProfileType.AUDIO_PROFILE_DEFAULT,
        audio_scenario=AudioScenarioType.AUDIO_SCENARIO_AI_SERVER,
        is_publish_audio=True, is_publish_video=False,
        audio_publish_type=AudioPublishType.AUDIO_PUBLISH_TYPE_PCM,
        video_publish_type=VideoPublishType.VIDEO_PUBLISH_TYPE_NONE)
    conn = svc.create_rtc_connection(conn_cfg, pub_cfg)
    conn.register_observer(Obs())
    conn.connect(creds["token"], args.channel, str(args.uid))
    lu = conn.get_local_user()
    lu.set_playback_audio_frame_before_mixing_parameters(1, RATE)
    conn.register_audio_frame_observer(Rec(), 0, None)
    conn.publish_audio()
    time.sleep(2)  # let the orchestrator see us join

    # publish the question in real time (20 ms chunks), then continuous silence —
    # endpointing only triggers on *received* silence
    marks["t_start"] = time.monotonic()
    chunk = 320 * 2
    for off in range(0, len(question_pcm), chunk):
        conn.push_audio_pcm_data(bytearray(question_pcm[off:off + chunk]), RATE, 1)
        time.sleep(0.02)

    t_last_downlink = None
    last_n = 0
    last_change = None
    while True:
        conn.push_audio_pcm_data(bytearray(b"\x00" * 640), RATE, 1)
        time.sleep(0.02)
        if len(captured) != last_n:
            last_n = len(captured)
            last_change = time.monotonic()
        if marks["downlink_started"] and last_n > 0 and last_change and \
           time.monotonic() - last_change > 3.0:
            print("[bot] reply finished (3 s without new downlink frames)")
            break
        if time.monotonic() - marks["t_start"] > 45:
            print("[bot] timeout while waiting for reply")
            break

    dur = len(captured) * 640 / (RATE * 2)
    latency = None
    if marks["downlink_started"]:
        latency = marks["downlink_started"] - marks["t_start"]
    print(f"[bot] downlink: {len(captured)} frames = {dur:.1f}s audio")
    print(f"[bot] latency question-start -> first downlink frame: "
          f"{latency if latency is not None else -1:.2f}s")

    if not captured:
        print("[bot] RESULT: FAIL (no downlink audio)")
        conn.disconnect(); conn.release(); svc.release()
        return 1

    # save + batch-ASR the captured reply
    wav_path = "/tmp/orch_reply.wav"
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(captured))
    dg = requests.post(
        "https://api.deepgram.com/v1/listen?model=nova-3&language=multi&sample_rate=16000&encoding=linear16&channels=1",
        headers={"Authorization": f"Token {env['DEEPGRAM_API_KEY']}",
                 "Content-Type": "audio/wav"},
        data=open(wav_path, "rb").read(),
        timeout=30)
    transcript = ""
    try:
        d = dg.json()
        alts = d["results"]["channels"][0]["alternatives"]
        transcript = alts[0]["transcript"] if alts else ""
    except Exception as e:
        print(f"[bot] batch ASR failed: {e}")
    print(f"[bot] reply transcript: {transcript!r}")

    ok = dur > 1.0 and bool(transcript.strip())
    print("E2E:", "PASS" if ok else "FAIL")
    conn.disconnect()
    conn.release()
    svc.release()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())