"""
Self-hosted voice session (Fases 3-5): the full VPS-hosted avatar loop.

    Agora channel (SD-RTN — transport unchanged)
      uplink (device uid 1 / web uid 2+), decoded PCM frames
        -> DeepgramStreamer nova-3, endpointing 500 ms            [Fase 2]
        -> utterance
        -> POST localhost /api/voice/chat-completions?avatar=X&streaming=1
           (RAG + sensors + species + weather + keyword modes: same brain) [Fase 3]
        -> SSE sentence deltas
        -> Cartesia sonic-3 (.ai) per-sentence TTS (prefetched, pipelined) [Fase 4]
        -> pure-Python G722 encoder (160 B per 320-sample frame)  [Fase 1]
        -> push_audio_encoded_data, absolute-clock paced 20 ms    [PoC-validated]

Audio path model (2026-09-12, from first live feedback):
- Playback is driven by an ABSOLUTE schedule (frame N at t0 + N*20 ms), not
  sleep(0.02)-after-push — drift made speech slow and jittery.
- TTS is PIPELINED: a worker fetches sentence N+1 while N is being pushed;
  inter-sentence gaps were heard as jitter.
- Barge-in (full-duplex, web sessions with AEC): uplink keeps feeding ASR
  while the avatar speaks; a new utterance bumps the turn generation —
  stale sentences/frames are dropped at 20 ms granularity.
- Half-duplex (device, no AEC): uplink discarded during downlink, with a
  tail-silence reopen delay. Transport-agnostic lesson #2.
Auto-idle ends the session after idle_timeout_s without user speech.
"""
import asyncio
import json
import array
import queue
import struct
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
                 language="multi", full_duplex=False, codec="g722", transport="agora",
                 rtp_port=26000, ws_port=8010, log=print):
        self.avatar_id = str(avatar_id)
        self.channel = channel
        self.token = token
        self.app_id = app_id
        self.dg_key = deepgram_key
        self.cart_key = cartesia_key
        self.tts_voice_id = tts_voice_id
        self.backend = backend_base
        self.language = language
        self.full_duplex = full_duplex   # web (AEC) -> True; device -> False
        self.codec = codec               # "g722" (device) | "opus" (web)
        self.transport = transport       # "agora" (SDK) | "rtp" (UDP direct)
        self.rtp_port = rtp_port         # VPS listen port (rtp mode)
        self.ws_port = ws_port           # VPS listen port (ws mode)
        self._ws_clients = set()         # live websocket clients
        self._device_addr = None         # learned from first received packet
        self._rtp_stats = {"pkts": 0, "samples": 0}
        self.log = log
        self.messages = []               # OpenAI-format turns (backend injects system prompt)

        self._uplink_q = queue.Queue()
        self._sentence_q = queue.Queue() # (gen, text)     -> tts worker
        self._pcm_q = queue.Queue()      # (gen, g722 frames) -> publisher
        self._muted = threading.Event()
        self._stop = threading.Event()
        self._sdk_ready = threading.Event()
        self._gen = 0                    # barge-in generation
        self._last_push = 0.0
        self._push_busy = False
        self._last_activity = time.monotonic()
        self._turn_active = False
        self._enc = g722.G722Encoder()
        self._g722_dec = None
        self._opus = None
        if codec == "opus":
            import opuslib_next as op
            self._opus = op.Encoder(16000, 1, op.APPLICATION_AUDIO)
        if transport == "rtp":
            self._g722_dec = g722.G722Decoder()

    # ------------------------- SDK thread -------------------------
    def _sdk_thread(self):
        svc_cfg = AgoraServiceConfig()
        svc_cfg.app_id = self.app_id
        svc = AgoraService()
        assert svc.initialize(svc_cfg) == 0
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
                # decoded uplink PCM of remote users (before mixing).
                # Full-duplex: always feed (browser AEC removes our downlink).
                # Half-duplex: discard while the avatar speaks.
                if session.full_duplex or not session._muted.is_set():
                    session._uplink_q.put(bytes(frame.buffer))
                return 1

        conn.register_observer(Obs())
        conn.connect(self.token, self.channel, str(777))
        lu = conn.get_local_user()
        lu.set_playback_audio_frame_before_mixing_parameters(1, RATE)
        conn.register_audio_frame_observer(Rec(), 0, None)
        conn.publish_audio()
        self._conn = conn
        self._sdk_ready.set()
        self._publisher(conn)
        conn.disconnect()
        conn.release()

    # ------------------------- audio out (worker + publisher) -------------------------
    def _tts_worker(self):
        """Sentence text -> Cartesia PCM (prefetch pipeline)."""

        while not self._stop.is_set():
            try:
                gen, text = self._sentence_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue  # barged in — drop stale sentence
            _t0 = time.monotonic()
            pcm = self._tts(text)
            if pcm and gen == self._gen:
                self.log(f"[TIMING] tts latency: {time.monotonic() - _t0:.2f}s "
                         f"({len(pcm) // 32000:.1f}s audio)")
            if self.transport in ("rtp", "ws") and pcm and gen == self._gen:
                chunks = [pcm[i:i + 640] for i in range(0, len(pcm) - 639, 640)]
                if len(pcm) % 640:
                    chunks.append(pcm[-(len(pcm) % 640):].ljust(640, b"\x00"))
                self._pcm_q.put((gen, chunks))
                continue
                if len(pcm) % (2 * 320):
                    pcm += b"\x00" * (640 - len(pcm) % 640)
                samples = array.array("h")
                samples.frombytes(pcm)
                if self._opus is not None:
                    # one opus packet per 20 ms frame — packets are
                    # variable-length and MUST be pushed individually
                    frames = [self._opus.encode(bytes(samples[k:k + 320]), 320)
                              for k in range(0, len(samples) - 319, 320)]
                else:
                    audio = self._enc.encode(samples)
                    frames = [audio[i:i + 160] for i in range(0, len(audio) - 159, 160)]
                self._pcm_q.put((gen, frames))
                self.log(f"[orch] tts ok gen{gen} ({len(pcm)} B): {text[:50]!r}")

    def _publisher(self, conn):
        """Frame pump on an ABSOLUTE 20 ms schedule (drift-free playback)."""
        codec_type = (AudioCodecType.AUDIO_CODEC_OPUS if self._opus is not None
                      else AudioCodecType.AUDIO_CODEC_G722)
        frame_info = EncodedAudioFrameInfo(
            codec=codec_type, sample_rate=RATE,
            samples_per_channel=FRAME_SAMPLES, number_of_channels=1,
            send_even_if_empty=1)
        threads = [threading.Thread(target=self._tts_worker, daemon=True)]
        for t in threads:
            t.start()
        while not self._stop.is_set():
            try:
                gen, frames = self._pcm_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue  # barged in
            try:
                next_t = time.monotonic()
                pushed = 0
                self._push_busy = True
                for f in frames:
                    if self._stop.is_set() or gen != self._gen:
                        break  # barged mid-sentence — stop at 20 ms granularity
                    conn.push_audio_encoded_data(f, frame_info)
                    pushed += 1
                    next_t += 0.02
                    delay = next_t - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                self._push_busy = False
                self._last_push = time.monotonic()
            except Exception as e:
                self.log(f"[orch] publisher error: {e}")

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

    def _speaking(self):
        if not self._pcm_q.empty() or not self._sentence_q.empty():
            return True
        return (time.monotonic() - self._last_push) < TAIL_SILENCE_S

    # ------------------------- turns -------------------------
    def _on_utterance(self, text):
        self._last_activity = time.monotonic()
        if not self.full_duplex:
            self._muted.set()   # half-duplex: gate uplink while we answer
        speaking = self._speaking() or self._turn_active
        if speaking:
            # BARGE-IN: bump generation — publisher drops current sentence at
            # 20 ms granularity, stale TTS/discarded deltas are ignored.
            self.log(f"[orch] BARGE-IN: {text!r}")
            self._gen += 1
            self._drain(self._sentence_q)
            self._drain(self._pcm_q)
            self._last_push = 0.0
        self._push_busy = False
        if self._turn_active:
            return  # previous turn thread cleans up; this utterance handled next
        self._turn_active = True
        threading.Thread(target=self._run_turn, args=(text, self._gen), daemon=True).start()

    def _drain(self, q):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _run_turn(self, text, gen):
        t0 = time.monotonic()
        first_delta_t = None
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
                if gen != self._gen:
                    reply_parts = []  # barged in mid-stream — discard
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                delta = ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content")
                if delta:
                    if first_delta_t is None:
                        first_delta_t = time.monotonic()
                        self.log(f"[TIMING] first LLM sentence ready: "
                                 f"{first_delta_t - t0:.2f}s")
                    self._sentence_q.put((gen, delta))
                    reply_parts.append(delta)
            resp.close()
            reply = "".join(reply_parts).strip()
            if reply and gen == self._gen:
                self.messages.append({"role": "assistant", "content": reply})
            self.log(f"[TIMING] utterance->stream done: {time.monotonic() - t0:.2f}s; "
                     f"reply {len(reply)} chars (gen {gen})")
        except Exception as e:
            self.log(f"[orch] LLM turn failed: {e}")
        finally:
            # half-duplex: reopen uplink once the PUSH SCHEDULE finishes (the
            # queue may still hold text being fetched/encoded — but audio
            # already queued continues pushing; waiting for full drain left
            # seconds of dead mic on long replies, felt as "slow to respond").
            if not self.full_duplex:
                last_reply = getattr(self, "_reply_gen", None)
                # wait for this turn's last sentence to be ENCODED (not played):
                while self._sentence_q.qsize() > 0 or self._pcm_q.qsize() > 0:
                    time.sleep(0.05)
                # wait for the publisher to finish PLAYING what it holds
                while self._push_busy:
                    time.sleep(0.05)
                self._muted.clear()
            self._turn_active = False

# ------------------------- RTP transport (Fase 6) -------------------------
    def _run_rtp(self, idle_timeout_s, max_duration_s):
        """Device talks raw UDP/RTP G722 directly to us — no Agora at all."""
        import socket
        self.log(f"[rtp] binding 0.0.0.0:{self.rtp_port} (device connects from its "
                 f"source addr; NAT hole via its own packets)")
        self._stop_publish = threading.Event()

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.rtp_port))
        sock.settimeout(0.2)
        self._rtp_sock = sock

        t_tts = threading.Thread(target=self._tts_worker, daemon=True)
        t_tts.start()
        t_pub = threading.Thread(target=self._rtp_publisher, args=(sock,), daemon=True)
        t_pub.start()

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

        self.log(f"[orch] live (rtp): avatar={self.avatar_id} channel={self.channel} "
                 f"mode={'full-duplex' if self.full_duplex else 'half-duplex'} "
                 f"codec=g722 idle={idle_timeout_s}s")
        self.log(f"[rtp] SPEAK to the sculpture — downlink flows once its addr is known")
        t_start = time.monotonic()
        self._gen = 0
        while True:
            now = time.monotonic()
            if max_duration_s and now - t_start > max_duration_s:
                self.log("[rtp] max duration reached — ending session")
                break
            if (not self._turn_active and now - self._last_activity > idle_timeout_s
                    and not self._speaking()):
                self.log("[rtp] auto-idle — ending session")
                break
            # uplink: RTP packet -> G722 payload -> decode -> PCM -> ASR
            try:
                pkt, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            if len(pkt) < 12:
                continue
            if self._device_addr is None:
                self._device_addr = addr
                self.log(f"[rtp] device registered from {addr[0]}:{addr[1]}")
            elif addr != self._device_addr:
                continue  # only the registered device
            # half-duplex gate: discard uplink while the avatar speaks
            if not self.full_duplex and self._muted.is_set():
                continue
            self._last_activity = max(self._last_activity, time.monotonic())
            payload = self._rtp_payload(pkt)
            pt = pkt[1] & 0x7F
            if pt == 96:
                pcm = self._g722_dec.decode(payload)
                audio = struct.pack("<%dh" % len(pcm), *pcm)
            else:
                audio = payload  # PT 97: raw s16le PCM
            self._rtp_stats["pkts"] += 1
            self._rtp_stats["samples"] += len(audio)
            if self._rtp_stats["pkts"] % 50 == 0:
                self.log(f"[rtp] rx {self._rtp_stats['pkts']} pkts, "
                         f"{self._rtp_stats['samples']} samples decoded, "
                         f"last payload {len(payload)} B")
            asyncio.run_coroutine_threadsafe(streamer.send_audio(audio), loop)

        asyncio.run_coroutine_threadsafe(streamer.close(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        self._stop.set()
        sock.close()
        self.log("[rtp] session ended")

    def _rtp_publisher(self, sock):
        """Encoded frames -> RTP packets -> device (absolute 20 ms schedule)."""
        self._rtp_seq = 0
        self._rtp_ts = 0
        while not self._stop.is_set():
            try:
                gen, frames = self._pcm_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != self._gen or self._device_addr is None:
                continue
            try:
                next_t = time.monotonic()
                self._push_busy = True
                for f in frames:
                    if self._stop.is_set() or gen != self._gen:
                        break
                    sock.sendto(self._rtp_packet(f, self._rtp_seq, self._rtp_ts),
                                self._device_addr)
                    self._rtp_seq = (self._rtp_seq + 1) & 0xFFFF
                    self._rtp_ts = (self._rtp_ts + 320) & 0xFFFFFFFF
                    next_t += 0.02
                    delay = next_t - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                self._last_push = time.monotonic()
            except Exception as e:
                self.log(f"[rtp] publisher error: {e}")
            finally:
                self._push_busy = False

    @staticmethod
    def _rtp_payload(pkt):
        """Strip the 12-byte RTP header (+ optional CSRC/extensions) -> payload."""
        b0 = pkt[0]
        version = b0 >> 6
        if version != 2:
            return pkt  # not RTP — treat as raw payload (device keepalive etc.)
        cc = b0 & 0x0F
        header_len = 12 + 4 * cc
        if len(pkt) > header_len and (pkt[0] & 0x10):  # extension bit
            ext_len = struct.unpack(">H", pkt[header_len + 2:header_len + 4])[0]
            header_len += 4 + 2 * ext_len
        return pkt[header_len:]

    def _rtp_packet(self, payload, seq, ts):
        """12-byte RTP header (incl. SSRC), PT 96 (dynamic G722), no CSRC."""
        b0 = 0x80  # version 2, no padding, no extension, cc=0
        return struct.pack("!BBHII", b0, 97, seq & 0xFFFF, ts, 0x77700001) + payload  # PT 97 = PCM

    # ------------------------- session loop -------------------------
    def run(self, idle_timeout_s=IDLE_TIMEOUT_S, max_duration_s=MAX_DURATION_S):
        if self.transport == "rtp":
            self._run_rtp(idle_timeout_s=idle_timeout_s, max_duration_s=max_duration_s)
            return
        if self.transport == "ws":
            self._run_ws(idle_timeout_s=idle_timeout_s, max_duration_s=max_duration_s)
            return
        self._sdk_ready = threading.Event()
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
                 f"mode={'full-duplex' if self.full_duplex else 'half-duplex'} "
                 f"codec={self._opus and 'opus' or 'g722'} "
                 f"idle={idle_timeout_s}s")
        t_start = time.monotonic()
        while True:
            now = time.monotonic()
            if max_duration_s and now - t_start > max_duration_s:
                self.log("[orch] max duration reached — ending session")
                break
            if (not self._turn_active and now - self._last_activity > idle_timeout_s
                    and not self._speaking()):
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
        self._stop.set()
        t_sdk.join(timeout=10)
        self.log("[orch] session ended")

# --- appended: WS transport methods (kept as module-level patch for review) ---

def _ws_patch(cls):
    """Attach WS transport methods to VoiceSession (Fase 6: web + device)."""
    def _run_ws(self, idle_timeout_s, max_duration_s):
        """WebSocket transport: binary frames = PCM s16le 16 kHz, both ways."""
        import asyncio
        import websockets
        self._stop_publish = threading.Event()
        self._ws_loop = asyncio.new_event_loop()
        t_loop = threading.Thread(
            target=lambda: (asyncio.set_event_loop(self._ws_loop),
                            self._ws_loop.run_forever()), daemon=True)
        t_loop.start()

        streamer = DeepgramStreamer(
            self.dg_key, language=self.language,
            on_interim=lambda t: setattr(self, "_last_activity", time.monotonic()),
            on_utterance=self._on_utterance)

        async def ws_handler(ws):
            self._ws_clients.add(ws)
            self.log(f"[ws] client connected ({len(self._ws_clients)} live)")
            try:
                async for msg in ws:
                    if not isinstance(msg, (bytes, bytearray)):
                        continue
                    if not self.full_duplex and self._muted.is_set():
                        continue
                    self._last_activity = max(self._last_activity, time.monotonic())
                    await streamer.send_audio(bytes(msg))
            except websockets.ConnectionClosed:
                pass
            except Exception as e:
                self.log(f"[ws] handler error: {e}")
            finally:
                self._ws_clients.discard(ws)
                self.log(f"[ws] client disconnected ({len(self._ws_clients)} live)")

        async def _start():
            self._ws_server = await websockets.serve(
                ws_handler, "0.0.0.0", self.ws_port, max_size=None)
            await streamer.connect()
        asyncio.run_coroutine_threadsafe(_start(), self._ws_loop).result(20)

        t_tts = threading.Thread(target=self._tts_worker, daemon=True)
        t_tts.start()
        t_pub = threading.Thread(target=self._ws_publisher, daemon=True)
        t_pub.start()

        self.log(f"[orch] live (ws): avatar={self.avatar_id} channel={self.channel} "
                 f"mode={'full-duplex' if self.full_duplex else 'half-duplex'} "
                 f"codec=pcm idle={idle_timeout_s}s")
        t_start = time.monotonic()
        while True:
            now = time.monotonic()
            if max_duration_s and now - t_start > max_duration_s:
                self.log("[ws] max duration reached — ending session")
                break
            if (not self._turn_active and now - self._last_activity > idle_timeout_s
                    and not self._speaking()):
                self.log("[ws] auto-idle — ending session")
                break
            time.sleep(0.2)

        async def _stop_all():
            self._ws_server.close()
            await self._ws_server.wait_closed()
            await streamer.close()
        asyncio.run_coroutine_threadsafe(_stop_all(), self._ws_loop).result(15)
        self._ws_loop.call_soon_threadsafe(self._ws_loop.stop)
        self._stop_publish.set()
        self.log("[ws] session ended")

    def _ws_publisher(self):
        """TTS PCM chunks -> binary frames broadcast to ws clients."""
        while not self._stop_publish.is_set():
            try:
                gen, chunks = self._pcm_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue
            try:
                self._push_busy = True
                pcm = b"".join(chunks)
                pcm = pcm[:len(pcm) - len(pcm) % 2]
                if pcm:
                    asyncio.run_coroutine_threadsafe(
                        self._ws_broadcast(pcm), self._ws_loop)
                self._last_push = time.monotonic()
            except Exception as e:
                self.log(f"[ws] publisher error: {e}")
            finally:
                self._push_busy = False

    async def _ws_broadcast(self, pcm):
        if not self._ws_clients:
            return
        import asyncio
        clients = [c for c in list(self._ws_clients) if c.state.name == "OPEN"]
        if clients:
            await asyncio.gather(*[c.send(pcm) for c in clients],
                                 return_exceptions=True)

    cls._run_ws = _run_ws
    cls._ws_publisher = _ws_publisher
    cls._ws_broadcast = _ws_broadcast


_ws_patch(VoiceSession)
