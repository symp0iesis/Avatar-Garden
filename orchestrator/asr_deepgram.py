"""
Fase 2 of the self-hosted orchestrator: Deepgram nova-3 streaming ASR
(WebSocket) with utterance endpointing.

Interface the orchestrator uses:
    streamer = DeepgramStreamer(api_key, on_utterance=cb, language="multi")
    await streamer.connect()
    await streamer.send_audio(pcm_chunk)   # s16le 16 kHz mono, from the RTC uplink
    ...
    await streamer.close()

on_utterance(full_text) fires when Deepgram signals end-of-utterance
(silence endpointing, ~500 ms). Interim transcripts are emitted via
on_interim for live display/logging only.
"""

import asyncio
import json
from urllib.parse import quote

import websockets

DEEPGRAM_WS = "wss://api.deepgram.com/v1/listen"

# Config mirrors the ConvoAI payload (nova-3, language=multi) plus the
# endpointing the orchestrator needs (~500 ms silence window, per Fase 2 plan).
# utterance_end_ms min step is 1000 (500 rejected by the API) — kept as a
# fallback signal; primary endpointing is endpointing=500 (speech_final).
DEFAULT_PARAMS = {
    "model": "nova-3",
    "language": "multi",
    "sample_rate": 16000,
    "encoding": "linear16",
    "channels": 1,
    "interim_results": "true",
    "endpointing": "500",        # silence window for endpointing (ms)
    "utterance_end_ms": "1000",  # UtteranceEnd message fallback after silence (ms)
    "punctuate": "false",
    "smart_format": "false",
}


class DeepgramStreamer:
    def __init__(self, api_key, on_utterance=None, on_interim=None,
                 language="multi", keyterms=None, endpointing_ms=500,
                 utterance_end_ms=1000):
        self._key = api_key
        self._on_utterance = on_utterance
        self._on_interim = on_interim
        params = dict(DEFAULT_PARAMS)
        params["language"] = language
        params["endpointing"] = str(endpointing_ms)
        params["utterance_end_ms"] = str(utterance_end_ms)
        query = "&".join(f"{k}={v}" for k, v in params.items())
        if keyterms:
            for kt in keyterms:
                query += f"&keyterm={quote(kt)}"
        self._ws_url = f"{DEEPGRAM_WS}?{query}"
        self._ws = None
        self._rx_task = None
        self._keepalive_task = None
        self._utterance_parts = []   # final segments of the current utterance
        self._open = False
        self._lock = asyncio.Lock()

    async def connect(self):
        headers = {"Authorization": f"Token {self._key}"}
        try:
            self._ws = await websockets.connect(self._ws_url, additional_headers=headers, max_size=None)
        except TypeError:
            # legacy websockets (<12) kwarg
            self._ws = await websockets.connect(self._ws_url, extra_headers=headers, max_size=None)
        self._open = True
        self._rx_task = asyncio.create_task(self._receive_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        return self

    async def send_audio(self, pcm_bytes):
        """Push s16le 16 kHz mono PCM chunk (e.g. 320-sample/20ms frames)."""
        if self._open and self._ws is not None:
            await self._ws.send(pcm_bytes)

    async def _keepalive_loop(self):
        try:
            while self._open:
                await asyncio.sleep(5)
                if self._open and self._ws is not None:
                    await self._ws.send('{"type": "KeepAlive"}')
        except asyncio.CancelledError:
            pass

    async def _receive_loop(self):
        try:
            async for raw in self._ws:
                self._handle_message(raw)
                if not self._open:
                    break
        except websockets.ConnectionClosed:
            pass

    def _handle_message(self, raw):
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        mtype = msg.get("type")

        if mtype == "Results":
            alt = (msg.get("channel") or {}).get("alternatives") or [{}]
            transcript = (alt[0] or {}).get("transcript", "")
            is_final = bool(msg.get("is_final"))
            speech_final = bool(msg.get("speech_final"))
            if transcript:
                if is_final:
                    self._utterance_parts.append(transcript)
                if self._on_interim:
                    live = " ".join(self._utterance_parts + ([] if is_final else [transcript]))
                    self._on_interim(live)
            # Primary endpointing signal: Deepgram saw endpointing ms of silence
            if speech_final:
                self._finish_utterance()
        elif mtype == "UtteranceEnd":
            # Fallback endpointing (utterance_end_ms)
            self._finish_utterance()
        elif mtype == "UtteranceEnd":
            self._finish_utterance()

    def _finish_utterance(self):
        text = " ".join(self._utterance_parts).strip()
        self._utterance_parts = []
        if text and self._on_utterance:
            self._on_utterance(text)

    async def close(self):
        self._open = False
        if self._keepalive_task:
            self._keepalive_task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._rx_task:
            self._rx_task.cancel()