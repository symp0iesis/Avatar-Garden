"""
Self-hosted voice session (Fases 3+4): the full VPS-hosted avatar loop.

    Agora channel (SD-RTN — transport unchanged)
      uplink (device uid 1 / web uid 2+), decoded PCM frames
        -> half-duplex gate (discarded while the avatar speaks)
        -> DeepgramStreamer nova-3, endpointing 500 ms            [Fase 2]
        -> utterance
        -> POST localhost /api/voice/chat-completions?avatar=X&streaming=1
           (RAG + sensors + species + weather + keyword modes: same brain) [Fase 3]
        -> SSE sentence deltas
        -> Cartesia sonic-3 (.ai) per-sentence TTS                [Fase 4]
        -> pure-Python G722 encoder (160 B per 320-sample frame)  [Fase 1]
        -> push_audio_encoded_data, paced 20 ms                   [PoC-validated]

Half-duplex: uplink muted while the avatar speaks (AIVAD is ConvoAI-side;
this is ours — transport-agnostic lesson #2). Auto-idle ends the session
after idle_timeout_s without user speech (native cost-guard).
"""
import asyncio
import array
import json
import queue
import threading
import time

import requests

import g722
from asr_deepgram import DeepgramStreamer

from agora.rtc.agora_service import AgoraService, AgoraServiceConfig
from agora.rtc.rtc_connection import RTCConnConfig, RtcConnectionPublishConfig
from agora.rtc.rtc_connection_observer import IRTCConnectionObserver
from agora.rtc.agora_base import (
    AudioProfileType, AudioScenarioType, AudioPublishType, VideoPublishType,
    AudioSubscriptionOptions, ClientRoleType, ChannelProfileType,
    AudioCodecType, EncodedAudioFrameInfo,
)
from agora.rtc.audio_frame_observer import IAudioFrameObserver

RATE = 16000
FRAME_SAMPLES = 320            # 20 ms @ 16 kHz
FRAME_BYTES = 160              # G722 64 kbps
ORCH_UID = 777
CARTESIA_TTS = "https://api.cartesia.ai/tts/bytes"
TAIL_SILENCE_S = 0.35          # half-duplex reopen delay after last downlink frame
IDLE_TIMEOUT_S = 180
MAX_DURATION_S = 600


class VoiceSession:
    def __init__(self, avatar_id, channel, token, app_id,
                 deepgram_key, cartesia_key, tts_voice_id,
                 backend_base="http://127.0.0.1:5001",
                 language="multi", log=print):
        self.avatar_id = str(avatar_id)
        self.channel = channel
        self.token = token
        self.app_id = app_id
        self.dg_key = deepgram_key
        self.cart_key = cartesia_key
        self.tts_voice_id = tts_voice_id
        self.backend = backend_base
        self.language = language
        self.log = log
        self.messages = []            # OpenAI-format turns (no system prompt: backend injects it)

        self._uplink_q = queue.Queue()
        self._tts_q = queue.Queue()   # text sentences -> publisher thread
        self._muted = threading.Event()
        self._stop_publish = threading.Event()
        self._sdk_ready = threading.Event()
        self._last_push = 0.0
        self._last_activity = time.monotonic()
        self._turn_active = False
        self._enc = g722.G722Encoder()
        self._conn = None
        self._svc = None

    # ------------------------- SDK thread -------------------------
    def _sdk_thread(self):
        svc_cfg = AgoraServiceConfig()
        svc_cfg.app_id = self.app_id
        svc = AgoraService()
        assert svc.initialize(svc_cfg) == 0
        self._svc = svc
        conn_cfg = RTCConnConfig(
            client_role_type=ClientRoleType.CLIENT_ROLE_BROADCASTER,
            channel_profile=ChannelProfileType.CHANNEL_PROFILE_LIVE_BROADCASTING,
            auto_subscribe_audio=1, auto_subscribe_video=0,
            audio_recv_media_packet=0,
            audio_subs_options=AudioSubscriptionOptions(
                packet_only=0, pcm_data_only=1, bytes_per_sample=2,
                number_of_channels=1, sample_rate_hz=RATE))
        pub_cfg = RtcConnectionPublishConfig(
            audio_profile=AudioProfileType.AUDIO_PROFILE_DEFAULT,
            audio_scenario=AudioScenarioType.AUDIO_SCENARIO_AI_SERVER,
            is_publish_audio=True, is_publish_video=False,
            audio_publish_type=AudioPublishType.AUDIO_PUBLISH_TYPE_ENCODED_PCM,
            video_publish_type=VideoPublishType.VIDEO_PUBLISH_TYPE_NONE)
        conn = svc.create_rtc_connection(conn_cfg, pub_cfg)

        session = self

        class Obs(IRTCConnectionObserver):
            def on_connected(self, *_a, **_k):
                session.log(f"[orch] connected to {session.channel}")
            def on_user_joined(self, conn, uid):
                session.log(f"[orch] user joined: {uid}")
            def on_user_left(self, conn, uid, reason):
                session.log(f"[orch] user left: {uid}")

        class Rec(IAudioFrameObserver):
            def on_playback_audio_frame_before_mixing(self, _l, _c, uid, frame, _v=0, _d=None):
                # decoded PCM of remote users (the uplink), before mixing
                if not session._muted.is_set():
                    session._uplink_q.put(bytes(frame.buffer))
                return 1

        conn.register_observer(Obs())
        conn.connect(self.token, self.channel, str(ORCH_UID))
        lu = conn.get_local_user()
        lu.set_playback_audio_frame_before_mixing_parameters(1, RATE)
        conn.register_audio_frame_observer(Rec(), 0, None)
        conn.publish_audio()
        self._conn = conn
        self._sdk_ready.set()

        # publisher: sentence text -> TTS -> G722 frames -> push (paced 20 ms)
        frame_info = EncodedAudioFrameInfo(
            codec=AudioCodecType.AUDIO_CODEC_G722, sample_rate=RATE,
            samples_per_channel=FRAME_SAMPLES, number_of_channels=1,
            send_even_if_empty=1)
        while not self._stop_publish.is_set():
            try:
                text = self._tts_q.get(timeout=0.2)
            except queue.Empty:
                continue
            pcm = self._tts(text)
            if not pcm:
                continue
            self.log(f"[orch] TTS ok ({len(pcm)} B): {text[:60]!r}")
            # pad to whole 20 ms frames, convert to int16 samples, encode
            if len(pcm) % (2 * FRAME_SAMPLES):
                pcm = pcm + b"\x00" * (640 - len(pcm) % 640)
            samples = array.array("h")
            samples.frombytes(pcm)
            audio = self._enc.encode(samples)
            for i in range(0, len(audio) - FRAME_BYTES + 1, FRAME_BYTES):
                if self._stop_publish.is_set():
                    break
                conn.push_audio_encoded_data(audio[i:i + FRAME_BYTES], frame_info)
                time.sleep(0.02)
            self._last_push = time.monotonic()
        conn.disconnect()
        conn.release()
        svc.release()

    def _tts(self, text):
        try:
            r = requests.post(
                CARTESIA_TTS,
                headers={"X-API-Key": self.cart_key, "Content-Type": "application/json",
                         "Cartesia-Version": "2026-08-14"},
                json={"transcript": text, "model_id": "sonic-3",
                      "voice": self.tts_voice_id,
                      "output_format": {"container": "raw", "encoding": "pcm_s16le",
                                        "sample_rate": RATE}},
                timeout=30)
            r.raise_for_status()
            return r.content
        except Exception as e:
            self.log(f"[orch] TTS failed: {e}")
            return b""

    # ------------------------- LLM turn -------------------------
    def _on_utterance(self, text):
        if self._turn_active:
            self.log(f"[orch] utterance dropped (turn in progress): {text!r}")
            return
        self.log(f"[orch] utterance: {text!r}")
        self._last_activity = time.monotonic()
        self._turn_active = True
        self._muted.set()
        threading.Thread(target=self._run_turn, args=(text,), daemon=True).start()

    def _run_turn(self, text):
        t0 = time.monotonic()
        try:
            self.messages.append({"role": "user", "content": text})
            resp = requests.post(
                f"{self.backend}/api/voice/chat-completions?avatar={self.avatar_id}&streaming=1",
                json={"messages": self.messages}, stream=True, timeout=120)
            resp.raise_for_status()
            reply_parts = []
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", "replace")
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                delta = ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content")
                if delta:
                    self._tts_q.put(delta)
                    reply_parts.append(delta)
            resp.close()
            reply = "".join(reply_parts).strip()
            if reply:
                self.messages.append({"role": "assistant", "content": reply})
            self.log(f"[TIMING] utterance->stream done: {time.monotonic() - t0:.2f}s; "
                     f"reply {len(reply)} chars")
        except Exception as e:
            self.log(f"[orch] LLM turn failed: {e}")
        finally:
            # wait until TTS publisher drains + tail, then reopen uplink
            while not self._tts_q.empty():
                time.sleep(0.05)
            time.sleep(max(0.0, TAIL_SILENCE_S - (time.monotonic() - self._last_push)))
            self._muted.clear()
            self._turn_active = False

    # ------------------------- session loop -------------------------
    def run(self, idle_timeout_s=IDLE_TIMEOUT_S, max_duration_s=MAX_DURATION_S):
        t_sdk = threading.Thread(target=self._sdk_thread, daemon=True)
        t_sdk.start()
        self._sdk_ready.wait(20)
        if self._conn is None:
            raise RuntimeError("SDK connection failed to initialize")

        streamer = DeepgramStreamer(
            self.dg_key, language=self.language,
            on_interim=lambda t: setattr(self, "_last_activity", time.monotonic()),
            on_utterance=self._on_utterance)
        loop = asyncio.new_event_loop()
        t_loop = threading.Thread(
            target=lambda: (asyncio.set_event_loop(loop), loop.run_forever()),
            daemon=True)
        t_loop.start()
        asyncio.run_coroutine_threadsafe(streamer.connect(), loop).result(15)

        self.log(f"[orch] live: avatar={self.avatar_id} channel={self.channel} "
                 f"idle={idle_timeout_s}s")
        t_start = time.monotonic()
        while True:
            now = time.monotonic()
            if max_duration_s and now - t_start > max_duration_s:
                self.log("[orch] max duration reached — ending session")
                break
            if (not self._turn_active and now - self._last_activity > idle_timeout_s
                    and not self._tts_busy()):
                self.log("[orch] auto-idle — ending session")
                break
            try:
                frame = self._uplink_q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._last_activity = max(self._last_activity, now)
            asyncio.run_coroutine_threadsafe(streamer.send_audio(frame), loop)

        asyncio.run_coroutine_threadsafe(streamer.close(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        self._stop_publish.set()
        t_sdk.join(timeout=10)
        self.log("[orch] session ended")

    def _tts_busy(self):
        if not self._tts_q.empty():
            return True
        return (time.monotonic() - self._last_push) < TAIL_SILENCE_S