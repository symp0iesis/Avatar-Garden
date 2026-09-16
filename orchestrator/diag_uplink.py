"""
Device uplink diagnosis: passive probe on the sculpture's channel.

Joins the device channel (uid 998, receive-only), captures N seconds of the
device's (uid 1) decoded uplink PCM, reports:
  - frame sizes + arrival rate + inter-arrival gaps (jitter)
  - RMS levels (speech/silence/clipping)
  - batch-ASR (Deepgram) of the captured audio -> transcript quality verdict

Run: /root/agora-orchestrator-venv/bin/python diag_uplink.py --channel marumbi-01
Then SPEAK a test sentence to the sculpture, e.g.:
  "Diagnostico de microfone: um, dois, tres, testando a voz do rio Marumbi."
"""
import argparse
import re
import struct
import subprocess
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
PROBE_UID = 998
DEVICE_UID = 1


def load_env():
    env = {}
    for line in open(f"{BACKEND}/.env"):
        m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="marumbi-01")
    ap.add_argument("--capture", type=float, default=14.0,
                    help="seconds to capture after first device frame")
    args = ap.parse_args()

    env = load_env()
    creds = requests.get(
        f"{BASE}/api/voice/token?channel={args.channel}&uid={PROBE_UID}").json()

    frames = []          # (t_monotonic, pcm_bytes)
    t_first = None

    class Rec(IAudioFrameObserver):
        def on_playback_audio_frame_before_mixing(self, _l, _c, uid, frame, _v=0, _d=None):
            nonlocal t_first
            if int(uid) == DEVICE_UID:
                now = time.monotonic()
                if t_first is None:
                    t_first = now
                    print(f"[probe] first device frame received", flush=True)
                frames.append((now, bytes(frame.buffer)))
            return 1

    class Obs(IRTCConnectionObserver):
        def on_connected(self, *_a, **_k):
            print(f"[probe] connected to {args.channel} (receive-only)", flush=True)
        def on_user_joined(self, conn, uid):
            print(f"[probe] user in channel: {uid}", flush=True)

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
        is_publish_audio=False, is_publish_video=False,
        audio_publish_type=AudioPublishType.AUDIO_PUBLISH_TYPE_PCM,
        video_publish_type=VideoPublishType.VIDEO_PUBLISH_TYPE_NONE)
    conn = svc.create_rtc_connection(conn_cfg, pub_cfg)
    conn.register_observer(Obs())
    conn.connect(creds["token"], args.channel, str(PROBE_UID))
    lu = conn.get_local_user()
    lu.set_playback_audio_frame_before_mixing_parameters(1, RATE)
    conn.register_audio_frame_observer(Rec(), 0, None)

    print(f"[probe] waiting for device (uid {DEVICE_UID}) audio... "
          f"SPEAK YOUR TEST SENTENCE NOW.", flush=True)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if t_first and frames and time.monotonic() - frames[-1][0] > 3.0:
            print("[probe] 3 s without new frames — capture complete")
            break
        if t_first and frames[-1][0] - t_first > args.capture:
            print(f"[probe] {args.capture}s captured")
            break
        time.sleep(0.1)

    if not frames:
        print("[probe] RESULT: FAIL — no device frames received. "
              "Is the sculpture powered and in the channel?")
        return 1

    # ---- frame stats ----
    sizes = [len(b) for _, b in frames]
    samples_per_frame = sizes[0] // 2
    gaps = [frames[i][0] - frames[i-1][0] for i in range(1, len(frames))]
    big_gaps = [g for g in gaps if g > 0.05]
    dur = frames[-1][0] - frames[0][0]
    print(f"\n[probe] frames: {len(frames)} over {dur:.2f}s "
          f"({len(frames)/max(dur,0.01):.0f} frames/s)")
    print(f"[probe] frame bytes: min={min(sizes)} max={max(sizes)} "
          f"(={samples_per_frame} samples/frame ≈ {RATE/1000*0.02*0.02:.0f}"
          f"{'ms' if samples_per_frame else ''} @16kHz if 640B)")
    print(f"[probe] inter-frame gaps: mean={sum(gaps)/len(gaps)*1000:.1f}ms "
          f"max={max(gaps)*1000:.0f}ms; gaps>50ms: {len(big_gaps)}")

    # ---- RMS levels (per ~1s chunk) ----
    import array
    pcm = b"".join(b for _, b in frames)
    arr = array.array("h")
    arr.frombytes(pcm[:len(pcm) // 2 * 2])
    chunk_n = RATE
    print("[probe] RMS per second (0=silence, ~3000+=speech, 32767=clipping):")
    for i in range(0, len(arr) - 1, chunk_n):
        ch = arr[i:i + chunk_n]
        rms = (sum(s * s for s in ch) / max(len(ch), 1)) ** 0.5
        peak = max(abs(s) for s in ch)
        print(f"  {i//RATE:3d}s: rms={rms:8.0f} peak={peak}")

    # ---- save WAV + batch ASR ----
    wav_path = "/tmp/uplink_diag.wav"
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    print(f"[probe] saved {wav_path} ({len(pcm)} B PCM = {len(pcm)/32000:.1f}s)")

    dg = requests.post(
        "https://api.deepgram.com/v1/listen?model=nova-3&language=multi&sample_rate=16000&encoding=linear16&channels=1",
        headers={"Authorization": f"Token {env['DEEPGRAM_API_KEY']}",
                 "Content-Type": "audio/wav"},
        data=pcm,
        timeout=30)
    try:
        d = dg.json()
        alts = d["results"]["channels"][0]["alternatives"]
        conf = alts[0]["confidence"] if alts else 0
        transcript = alts[0]["transcript"] if alts else ""
    except Exception as e:
        print(f"[probe] batch ASR failed: {e}")
        return 1
    print(f"[probe] ASR transcript (confidence {conf:.2f}): {transcript!r}")

    verdict = "CLEAN" if len(transcript) > 15 and conf > 0.7 else (
        "GARBLED — expect uplink fix (resampler/rate)" if transcript else "SILENT")
    print(f"[probe] UPLINK VERDICT: {verdict}")
    conn.disconnect()
    conn.release()
    svc.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())