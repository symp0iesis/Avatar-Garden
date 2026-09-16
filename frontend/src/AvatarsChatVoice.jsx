import { useState, useRef, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { Mic, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import AgoraRTC from "agora-rtc-sdk-ng";
import LatencyBreakdown from "./LatencyBreakdown";

AgoraRTC.setLogLevel(3); // warn only

const STATUS_LABEL = {
  idle: "Press to start conversation",
  connecting: "Connecting…",
  listening: "Listening…",
  thinking: "The river contemplates…",
  speaking: "The river speaks…",
  error: "Connection error",
};

export default function AvatarsChatVoice() {
  const [status, setStatus] = useState("idle");
  const [isConnected, setIsConnected] = useState(false);
  const [isDisconnecting, setIsDisconnecting] = useState(false);
  const [userVolume, setUserVolume] = useState(0);
  const [avatarVolume, setAvatarVolume] = useState(0);
  const [avatars, setAvatars] = useState([]);
  const [selectedAvatarId, setSelectedAvatarId] = useState(null);
  const [agentId, setAgentId] = useState(null);
  const [error, setError] = useState(null);

  const [latencyExpanded, setLatencyExpanded] = useState(false);
  const [lastLatency, setLastLatency] = useState(null); // { perceivedMs, timings }
  // Experimental: stream LLM sentences to Agora (TTS starts sooner). Off by
  // default — Agora-side chunk handling still being ironed out (ISSUES.md 14).
  const [streamingEnabled, setStreamingEnabled] = useState(false);

  const clientRef = useRef(null);
  const micTrackRef = useRef(null);
  const wsRef = useRef(null);            // VPS ws voice (Fase 6)
  const audioCtxRef = useRef(null);
  const playNextRef = useRef(0);
  const analyserRef = useRef(null);
  const micStreamRef = useRef(null);
  const startRef = useRef(null);
  const [isWs, setIsWs] = useState(false);
  const volumeRafRef = useRef(null);
  // Unique channel per session so multiple browser tabs don't collide
  const channelRef = useRef(`avatar-lab-${Date.now()}`);

  // --- latency measurement refs ---
  // Perceived latency = time from the user's last voice activity (≈ end of speech)
  // to the first agent audio of the next turn. The backend slice is fetched from
  // /api/voice/last-timings; the remainder is the Agora pipeline (ASR + TTS + transport).
  // KNOWN LIMITATION: ambient mic noise above the activity threshold keeps sliding
  // the anchor forward, which can shrink the perceived total below the backend time
  // and hide the derived Agora segment. A stricter hysteresis-based speech-end
  // detector was tried (2026-07-31) but failed to trigger at all in practice —
  // revisit with real mic-level data.
  const lastVoiceActivityRef = useRef(null);
  const agentSpeakingRef = useRef(false);
  const agentQuietSinceRef = useRef(null);
  const avatarIdRef = useRef(null);

  const onAgentTurnStart = async (now) => {
    const t0 = lastVoiceActivityRef.current;
    if (!t0) return;
    const perceivedMs = Math.round(now - t0);
    let timings = {};
    try {
      const resp = await fetch(`/api/voice/last-timings?avatar=${avatarIdRef.current}`);
      const data = await resp.json();
      // Only trust timings from a recent request (guards against stale entries)
      if (data.ts && Date.now() / 1000 - data.ts < 120) timings = data;
    } catch (_e) { /* show perceived total only */ }
    setLastLatency({ perceivedMs, timings });
  };
  const onAgentTurnStartRef = useRef(onAgentTurnStart);
  onAgentTurnStartRef.current = onAgentTurnStart;

  // Load avatar list on mount
  useEffect(() => {
    fetch("/api/avatars")
      .then(r => r.json())
      .then(data => {
        setAvatars(data);
        if (data.length > 0) setSelectedAvatarId(data[0].id);
      })
      .catch(e => console.error("Failed to load avatars:", e));
  }, []);

  // Poll mic volume for user ripple
  const startVolumePolling = useCallback((micTrack) => {
    const poll = () => {
      if (!micTrack) return;
      const v = micTrack.getVolumeLevel?.() ?? 0;
      // Track the user's last voice activity — used as the start point of
      // perceived latency when the agent's reply audio begins.
      if (v > 0.1) lastVoiceActivityRef.current = performance.now();
      setUserVolume(v);
      volumeRafRef.current = requestAnimationFrame(poll);
    };
    poll();
  }, []);

  const stopVolumePolling = useCallback(() => {
    if (volumeRafRef.current) {
      cancelAnimationFrame(volumeRafRef.current);
      volumeRafRef.current = null;
    }
    setUserVolume(0);
    setAvatarVolume(0);
  }, []);

  const disconnect = useCallback(async (silent = false) => {
    if (!silent) setIsDisconnecting(true);
    stopVolumePolling();

    if (agentId) {
      try {
        await fetch("/api/voice/agent/stop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ agentId }),
        });
      } catch (e) {
        if (!silent) console.error("Stop agent error:", e);
      }
      setAgentId(null);
    }

    if (micTrackRef.current) {
      micTrackRef.current.close();
      micTrackRef.current = null;
    }

    if (clientRef.current) {
      try { await clientRef.current.leave(); } catch (_e) { /* ignore */ }
      clientRef.current = null;
    }

    setIsConnected(false);
    setIsDisconnecting(false);
    setStatus("idle");
  }, [agentId, stopVolumePolling]);

  const disconnectWs = async () => {
    setIsDisconnecting(true);
    try { wsRef.current?.close(); } catch (_e) {}
    wsRef.current = null;
    micStreamRef.current?.getTracks().forEach(t => t.stop());
    micStreamRef.current = null;
    if (audioCtxRef.current) { try { await audioCtxRef.current.close(); } catch (_e) {} }
    audioCtxRef.current = null;
    setUserVolume(0); setAvatarVolume(0);
    setIsConnected(false); setIsWs(false);
    setIsDisconnecting(false); setStatus("idle");
    await fetch("/api/voice/agent/stop", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agentId: startRef.current }),
    }).catch(() => {});
    startRef.current = null;
  };

  const connectWs = async (start) => {
    setIsWs(true);
    startRef.current = start.agentId;
    setStatus("connecting");
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(proto + "//" + location.host + start.wsPath);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    audioCtxRef.current = ctx;
    if (ctx.state === "suspended") await ctx.resume();  // without this the mic capture never fires
    playNextRef.current = 0;

    let speakTimer = null;
    ws.onmessage = (e) => {
      if (!(e.data instanceof ArrayBuffer)) return;
      const pcm = new Int16Array(e.data);
      if (!pcm.length) return;
      setStatus("speaking");
      // RMS for the avatar orb
      let peak = 0;
      for (let i = 0; i < pcm.length; i += 16) peak = Math.max(peak, Math.abs(pcm[i]));
      setAvatarVolume(peak / 32768);
      // schedule playback (continuous stream)
      const buf = ctx.createBuffer(1, pcm.length, 16000);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
      const srcNode = ctx.createBufferSource();
      srcNode.buffer = buf;
      srcNode.connect(ctx.destination);
      const now = ctx.currentTime;
      if (playNextRef.current < now + 0.05) playNextRef.current = now + 0.05;
      srcNode.start(playNextRef.current);
      playNextRef.current += buf.duration;
      if (speakTimer) clearTimeout(speakTimer);
      speakTimer = setTimeout(() => {
        setAvatarVolume(0);
        setStatus("listening");
      }, 600);
    };
    ws.onerror = (e) => { console.error("ws error", e); setError("WebSocket error"); setStatus("error"); };
    await new Promise((res, rej) => {
      ws.onopen = res;
      ws.onclose = () => rej(new Error("voice websocket closed"));
    });

    // mic capture -> int16 PCM -> ws
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    micStreamRef.current = stream;
    const micSource = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    analyserRef.current = analyser;
    micSource.connect(analyser);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const mute = ctx.createGain();
    mute.gain.value = 0;  // mic must not reach output (AEC handles real echo)
    proc.onaudioprocess = (e) => {
      if (wsRef.current?.readyState !== WebSocket.OPEN) return;
      const f = e.inputBuffer.getChannelData(0);
      const int16 = new Int16Array(f.length);
      for (let i = 0; i < f.length; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(f[i] * 32767)));
      }
      wsRef.current.send(int16.buffer);
    };
    proc.connect(mute);
    mute.connect(ctx.destination);

    // user orb volume from the analyser
    const volData = new Uint8Array(analyser.frequencyBinCount);
    const poll = () => {
      if (!analyserRef.current) return;
      analyser.getByteTimeDomainData(volData);
      let peak = 0;
      for (let i = 0; i < volData.length; i++) peak = Math.max(peak, Math.abs(volData[i] - 128));
      setUserVolume(peak / 128);
      requestAnimationFrame(poll);
    };
    poll();

    setIsConnected(true);
    setStatus("listening");
  };

  const connect = async () => {
    if (!selectedAvatarId || status === "connecting") return;
    setError(null);
    setStatus("connecting");
    avatarIdRef.current = selectedAvatarId;

    try {
      // 0. Ask the backend to spawn the voice session (response selects transport)
      const startResp = await fetch("/api/voice/agent/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          avatarId: selectedAvatarId,
          channel: channelRef.current,
          userUid: 0,
          streaming: streamingEnabled,
        }),
      });
      if (!startResp.ok) {
        const err = await startResp.json();
        throw new Error(typeof err.error === "string" ? err.error : JSON.stringify(err.error));
      }
      const start = await startResp.json();

      if (start.transport === "ws") {
        return await connectWs(start);
      }
      // else: fall through to the Agora path with the session already spawned
      const tokenResp = await fetch(`/api/voice/token?channel=${channelRef.current}`);
      const { appId, token, channel, uid } = await tokenResp.json();

      if (!appId) throw new Error("AGORA_APP_ID is not configured on the backend.");

      // 2. Create Agora RTC client
      const client = AgoraRTC.createClient({ mode: "rtc", codec: "vp8" });
      clientRef.current = client;

      // When the AI agent publishes its audio response, play it
      client.on("user-published", async (user, mediaType) => {
        await client.subscribe(user, mediaType);
        if (mediaType === "audio") {
          user.audioTrack.play();
          setStatus("speaking");

          // Poll avatar track volume for ripple animation + turn-start detection
          const pollAvatar = () => {
            const v = user.audioTrack.getVolumeLevel?.() ?? 0;
            const now = performance.now();
            if (v > 0.08) {
              if (!agentSpeakingRef.current) {
                // Rising edge — the agent's reply audio just started
                agentSpeakingRef.current = true;
                onAgentTurnStartRef.current(now);
              }
              agentQuietSinceRef.current = null;
            } else if (agentSpeakingRef.current) {
              // Require >1s of silence before treating the turn as over,
              // so natural mid-sentence pauses don't retrigger detection
              if (agentQuietSinceRef.current == null) {
                agentQuietSinceRef.current = now;
              } else if (now - agentQuietSinceRef.current > 1000) {
                agentSpeakingRef.current = false;
                agentQuietSinceRef.current = null;
              }
            }
            setAvatarVolume(v);
            volumeRafRef.current = requestAnimationFrame(pollAvatar);
          };
          pollAvatar();
        }
      });

      client.on("user-unpublished", (_user, mediaType) => {
        if (mediaType === "audio") {
          setAvatarVolume(0);
          agentSpeakingRef.current = false;
          agentQuietSinceRef.current = null;
          setStatus("listening");
        }
      });

      client.on("user-left", () => {
        setAvatarVolume(0);
        setStatus("listening");
      });

      // Listen for connection state changes
      client.on("connection-state-change", (curState, revState, reason) => {
        console.log(`[Agora] Connection: ${revState} → ${curState}, reason: ${reason}`);
      });

      client.on("exception", (evt) => {
        console.error("[Agora] Exception:", evt);
      });

      // 3. Request mic access FIRST (so we fail fast before joining channel)
      console.log("[Agora] Requesting microphone access...");
      const micTrack = await AgoraRTC.createMicrophoneAudioTrack({
        encoderConfig: "speech_low_quality",
        AEC: true,
        ANS: true,
        AGC: true,
      });
      micTrackRef.current = micTrack;
      console.log("[Agora] Mic access granted");

      // 4. Join channel
      console.log("[Agora] Joining:", { appId, channel, token: token?.slice(0, 20) + "...", uid });
      const assignedUid = await client.join(appId, channel, token ?? null, uid || null);
      console.log("[Agora] Joined successfully, assigned uid:", assignedUid);

      // 5. Publish mic track
      console.log("[Agora] Publishing mic track...");
      await client.publish([micTrack]);
      console.log("[Agora] Mic track published");
      startVolumePolling(micTrack);

      // 6. Start the AI agent on the backend (which calls Agora's REST API)
      const agentResp = await fetch("/api/voice/agent/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          avatarId: selectedAvatarId,
          channel,
          userUid: uid || 0,
          streaming: streamingEnabled,
        }),
      });

      if (!agentResp.ok) {
        const err = await agentResp.json();
        throw new Error(typeof err.error === "string" ? err.error : JSON.stringify(err.error));
      }

      const { agentId: id } = await agentResp.json();
      setAgentId(id);
      setIsConnected(true);
      setStatus("listening");

    } catch (e) {
      console.error("Voice connection failed:", e);
      setError(e.message);
      setStatus("error");
      await disconnect(true);
    }
  };

  // Cleanup on unmount — use ref to avoid re-triggering on every render
  const disconnectRef = useRef(disconnect);
  disconnectRef.current = disconnect;
  useEffect(() => {
    return () => {
        if (isWs) { try { wsRef.current?.close(); } catch (_e) {} }
        disconnectRef.current(true);
      };
  }, []);

  const userRippleScale = 1 + userVolume * 2;
  const avatarRippleScale = 1 + avatarVolume * 2;
  const selectedAvatar = avatars.find(a => a.id === selectedAvatarId);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 gap-6">

      {/* Header */}
      <div className="text-center">
        <h1 className="text-2xl md:text-3xl font-poetic font-semibold tracking-tight text-garden-ink">Voice Avatar Garden</h1>
        <p className="font-poetic text-garden-inksoft text-sm mt-1">Live voice conversations with nature avatars</p>
      </div>

      {/* Session setup — avatar + streaming, shown when not connected */}
      {!isConnected && avatars.length > 0 && (
        <div className="w-full max-w-md rounded-xl border border-garden-line bg-card p-4 shadow-sm space-y-3">
          <div>
            <label className="block font-poetic text-garden-inksoft text-xs uppercase tracking-wider mb-1.5">Avatar</label>
            <select
              className="w-full p-2.5 rounded-md border border-garden-line bg-garden-paper font-poetic text-sm"
              value={selectedAvatarId ?? ""}
              onChange={e => setSelectedAvatarId(e.target.value)}
            >
              {avatars.map(a => (
                <option key={a.id} value={a.id}>{a.name}</option>
              ))}
            </select>
          </div>
          {/* Streaming toggle — ConvoAI sessions only (the VPS orchestrator always
              streams); applies to the next conversation start */}
          {!isConnected && selectedAvatar?.llmDefaults?.voiceBackend !== "vps" && (
          <div className="flex items-center justify-between p-3 rounded-lg border border-garden-line bg-garden-paper2">
            <div className="pr-3">
              <p className="font-poetic text-garden-ink text-sm font-medium">Streaming responses</p>
              <p className="font-poetic text-garden-inksoft text-xs mt-0.5">Experimental — speech starts sooner</p>
            </div>
            <button
              role="switch"
              aria-checked={streamingEnabled}
              onClick={() => setStreamingEnabled(!streamingEnabled)}
              className={"relative w-11 h-6 rounded-full transition-colors duration-200 shrink-0 " + (streamingEnabled ? "bg-garden-moss" : "bg-garden-line")}
            >
              <span className={"absolute top-0.5 left-0.5 w-5 h-5 bg-garden-paper rounded-full shadow transition-transform duration-200 " + (streamingEnabled ? "translate-x-5" : "translate-x-0")} />
            </button>
          </div>
          )}
        </div>
      )}

      {/* Live session badge */}
      {isConnected && selectedAvatar && (
        <div className="flex items-center gap-2 px-4 py-1.5 rounded-full border border-garden-line bg-card shadow-sm">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-garden-moss opacity-60"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-garden-moss"></span>
          </span>
          <span className="font-poetic text-sm text-garden-ink">
            Live — speaking with <span className="font-semibold">{selectedAvatar.name}</span>
          </span>
        </div>
      )}

      {/* Latency Analysis */}
      <div className="w-full max-w-md">
        <button
          className="flex items-center gap-2 font-poetic text-garden-inksoft cursor-pointer hover:text-garden-ink"
          onClick={() => setLatencyExpanded(!latencyExpanded)}
        >
          <span className="text-lg">{latencyExpanded ? '▼' : '▶'}</span>
          <span className="font-semibold">Latency Analysis</span>
          {lastLatency && (
            <span className="text-xs text-garden-inksoft font-data">
              {(lastLatency.perceivedMs / 1000).toFixed(2)}s
            </span>
          )}
        </button>

        {latencyExpanded && (
          <div className="mt-2 p-3 rounded-lg border bg-garden-paper2">
            {!lastLatency ? (
              <p className="font-poetic text-garden-inksoft text-sm">
                Complete a voice exchange to see its latency breakdown.
              </p>
            ) : (() => {
                const t = lastLatency.timings;
                // Streaming turns expose llm_first_token_ms; the panel must then
                // measure up to the FIRST token (matching the first-audio moment).
                // main_llm_ms covers the entire stream and would overstate latency.
                const isStreaming = (t.llm_first_token_ms || 0) > 0;
                const segments = [
                  { label: "Loading RAG index into RAM", ms: t.index_load_ms || 0, color: "#9333EA" },
                  { label: "Keyword generation (LLM)", ms: t.keyword_gen_ms || 0, color: "#CA8A04" },
                  { label: "Knowledge retrieval (RAG)", ms: t.rag_retrieval_ms || 0, color: "#0E7490" },
                  { label: "Web search", ms: t.web_search_ms || 0, color: "#DB2777" },
                  { label: "Sensor snapshot", ms: t.sensor_snapshot_ms || 0, color: "#15803D" },
                  isStreaming
                    ? { label: "LLM time to first token (streaming)", ms: t.llm_first_token_ms, color: "#1D4ED8" }
                    : { label: "Avatar response (LLM)", ms: t.main_llm_ms || 0, color: "#1D4ED8" },
                  { label: "Sensor analysis tool", ms: t.sensor_tool_ms || 0, color: "#EA580C" },
                ];
                const attributed = segments.reduce((acc, s) => acc + s.ms, 0);
                if (!isStreaming) {
                  const backendOther = Math.max((t.total_backend_ms || 0) - attributed, 0);
                  if (backendOther > 0) segments.push({ label: "Backend overhead", ms: backendOther, color: "#A8A29E" });
                }
                const agora = Math.max(lastLatency.perceivedMs - attributed, 0);
                if (agora > 0) segments.push({ label: isStreaming ? "TTS streaming + Agora pipeline (rest of reply)" : "Agora pipeline (ASR + TTS + transport)", ms: agora, color: "#475569" });
                return (
                  <>
                    <LatencyBreakdown segments={segments} totalMs={lastLatency.perceivedMs} />
                    <p className="mt-2 text-[10px] text-garden-inksoft font-poetic">
                      Measured from end of your speech to first avatar audio. Agora pipeline time is derived, not directly measured.
                    </p>
                  </>
                );
              })()}
          </div>
        )}
      </div>

      {/* Conversation orbs — side by side, flow connector between */}
      <div className="flex items-center gap-4 md:gap-8 my-6">

        {/* You */}
        <div className="flex flex-col items-center gap-3">
          <div className="relative flex items-center justify-center w-28 h-28 md:w-32 md:h-32">
            <div className="absolute inset-0 rounded-full border border-garden-line bg-card/60" />
            <motion.div
              className="absolute w-24 h-24 rounded-full bg-garden-moss/50"
              animate={{ scale: isConnected ? userRippleScale : 1, opacity: isConnected ? 0.55 : 0.12 }}
              transition={{ duration: 0.08 }}
            />
            <div className={"relative w-10 h-10 rounded-full transition-colors " + (isConnected ? "bg-garden-moss" : "bg-garden-line")} />
          </div>
          <span className="font-poetic text-xs text-garden-inksoft">You</span>
        </div>

        {/* Connector */}
        <div className="flex items-center w-14 md:w-24">
          {isConnected ? (
            <div className="flex items-center justify-between w-full px-1">
              {[0, 1, 2].map(i => (
                <motion.span
                  key={i}
                  className="w-1.5 h-1.5 rounded-full bg-garden-moss"
                  animate={{ opacity: [0.15, 1, 0.15], scale: [0.8, 1.2, 0.8] }}
                  transition={{ repeat: Infinity, duration: 1.4, delay: i * 0.25 }}
                />
              ))}
            </div>
          ) : (
            <div className="w-full border-t border-dashed border-garden-line" />
          )}
        </div>

        {/* Avatar */}
        <div className="flex flex-col items-center gap-3">
          <div className="relative flex items-center justify-center w-28 h-28 md:w-32 md:h-32">
            <div className="absolute inset-0 rounded-full border border-garden-line bg-card/60" />
            <motion.div
              className="absolute w-24 h-24 rounded-full bg-garden-water/50"
              animate={{
                scale: status === "speaking" ? avatarRippleScale : 1,
                opacity: status === "speaking" ? 0.55 : 0.12,
              }}
              transition={{ duration: 0.08 }}
            />
            <div className="relative w-10 h-10 rounded-full bg-garden-water/80" />
          </div>
          <span className="font-poetic text-xs text-garden-ink max-w-[10rem] truncate text-center">
            {selectedAvatar?.name || "Avatar"}
          </span>
        </div>
      </div>

      {/* Status */}
      <div className="flex items-center gap-2 min-h-[1.5rem]">
        <span className={"w-2 h-2 rounded-full " + (
          status === "listening" ? "bg-garden-moss animate-pulse"
          : status === "speaking" ? "bg-garden-water"
          : status === "thinking" || status === "connecting" ? "bg-garden-amber animate-pulse"
          : status === "error" ? "bg-garden-clay"
          : "bg-garden-line"
        )} />
        <p className="font-poetic text-garden-inksoft text-sm">
          {STATUS_LABEL[status] ?? ""}
        </p>
      </div>

      {/* Start / End button — always below the orbs */}
      {!isConnected ? (
        <Button
          onClick={connect}
          disabled={status === "connecting" || !selectedAvatarId}
          className="bg-garden-moss hover:bg-garden-mossdeep text-garden-paper rounded-md px-8 py-3 text-base font-poetic"
        >
          {status === "connecting" ? (
            "Connecting…"
          ) : (
            <span className="inline-flex items-center gap-2"><Mic size={18} /> Start Conversation</span>
          )}
        </Button>
      ) : (
        <Button
          onClick={() => (isWs ? disconnectWs() : disconnect())}
          disabled={isDisconnecting}
          className="bg-garden-clay hover:bg-garden-clay/80 text-garden-paper rounded-md px-8 py-3 text-base font-poetic disabled:opacity-60"
        >
          <span className="inline-flex items-center gap-2"><Square size={14} /> {isDisconnecting ? "Ending…" : "End Conversation"}</span>
        </Button>
      )}

      {/* Error display */}
      {error && (
        <p className="text-sm text-garden-clay max-w-sm text-center font-poetic">{error}</p>
      )}

      {/* Force quit — escape hatch for orphaned sessions */}
      <button
        className="text-xs text-garden-inksoft underline hover:text-garden-ink cursor-pointer"
        onClick={async () => {
          try {
            const r = await fetch("/api/voice/sessions/stop-all", { method: "POST" });
            const d = await r.json();
            setError(d.stopped?.length ? `Force quit: ${d.stopped.join(", ")}` : "Nothing live to quit");
          } catch (_e) { setError("Force quit failed"); }
        }}
      >
        force quit any live session
      </button>

      {/* Device integration guide */}
      {!isConnected && (
        <details className="mt-4 max-w-lg w-full">
          <summary className="text-xs text-garden-inksoft cursor-pointer font-poetic">
            Connecting a hardware device (Raspberry Pi / ReSpeaker)
          </summary>
          <div className="mt-2 text-xs text-garden-inksoft space-y-3 font-data">
            <p className="font-poetic text-garden-inksoft">Your device joins a shared voice channel. The avatar listens, thinks, and speaks back — your device only needs to handle audio input and output.</p>
            <p className="font-poetic text-garden-inksoft">
              Base URL: <code className="bg-garden-paper2 rounded px-1">https://avatars.sympoiesis.xyz</code>
            </p>

            <p className="font-poetic font-semibold text-garden-inksoft">Requirements</p>
            <p className="font-poetic">Python 3.10 or newer. Linux or macOS (Raspberry Pi / arm64 supported, Windows is not).</p>
            <pre className="bg-garden-paper2 rounded p-2 overflow-x-auto whitespace-pre-wrap">{`# On Raspberry Pi, install system audio libraries first:
sudo apt-get install python3-pyaudio portaudio19-dev

# Then install Python packages:
pip install agora-python-server-sdk pyaudio requests`}</pre>

            <p className="font-poetic font-semibold text-garden-inksoft">Step 0 — Find available avatars</p>
            <pre className="bg-garden-paper2 rounded p-2 overflow-x-auto whitespace-pre-wrap">{`GET /api/avatars
← [{ "id": "0", "name": "Lahn" }, { "id": "1", "name": "..." }, ...]`}</pre>

            <p className="font-poetic font-semibold text-garden-inksoft">Step 1 — Get a channel token</p>
            <p className="font-poetic">Choose a unique name for your device's channel and any numeric user ID.</p>
            <pre className="bg-garden-paper2 rounded p-2 overflow-x-auto whitespace-pre-wrap">{`GET /api/voice/token?channel=my-device&uid=1
← { appId, channel, uid, token }`}</pre>

            <p className="font-poetic font-semibold text-garden-inksoft">Step 2 — Start the avatar</p>
            <p className="font-poetic">Use the avatar <code className="bg-garden-paper2 rounded px-1">id</code> from Step 0 and the <code className="bg-garden-paper2 rounded px-1">channel</code> / <code className="bg-garden-paper2 rounded px-1">uid</code> from Step 1.</p>
            <pre className="bg-garden-paper2 rounded p-2 overflow-x-auto whitespace-pre-wrap">{`POST /api/voice/agent/start
Content-Type: application/json
{ "avatarId": "0", "channel": "my-device", "userUid": 1 }
← { agentId, channel }`}</pre>

            <p className="font-poetic font-semibold text-garden-inksoft">Step 3 — Run the client script</p>
            <p className="font-poetic">Save the script below as <code className="bg-garden-paper2 rounded px-1">avatar_client.py</code>. Edit the four constants at the top, then run it. It handles Steps 1 and 2 automatically, joins the channel, streams your mic to the avatar, and plays the avatar's voice through your speaker. Press Ctrl+C to end.</p>
            <pre className="bg-garden-paper2 rounded p-2 overflow-x-auto whitespace-pre-wrap">{`#!/usr/bin/env python3
import asyncio, pyaudio, requests
from agora.rtc.agora_service import AgoraService, AgoraServiceConfig
from agora.rtc.rtc_connection import RTCConnConfig, RtcConnectionPublishConfig
from agora.rtc.agora_base import (
    AudioProfileType, AudioScenarioType, AudioPublishType, VideoPublishType,
    AudioSubscriptionOptions, ClientRoleType, ChannelProfileType,
)
from agora.rtc.audio_frame_observer import IAudioFrameObserver

# ── configure these ──────────────────────────────────────────────
BASE    = "https://avatars.sympoiesis.xyz"
CHANNEL = "my-device"   # unique name for this device
UID     = 1             # any integer
AVATAR  = "0"           # avatar id from GET /api/avatars
# ─────────────────────────────────────────────────────────────────

RATE, CH, CHUNK = 16000, 1, int(16000 * 0.02)  # 16 kHz mono, 20 ms frames

pa      = pyaudio.PyAudio()
speaker = pa.open(format=pyaudio.paInt16, channels=CH, rate=RATE, output=True)

class SpeakerObserver(IAudioFrameObserver):
    def on_playback_audio_frame_before_mixing(self, _lu, _cid, _uid, frame, _vs=0, _vd=None):
        speaker.write(bytes(frame.buffer))
        return 1

async def main():
    creds = requests.get(f"{BASE}/api/voice/token?channel={CHANNEL}&uid={UID}").json()
    agent = requests.post(f"{BASE}/api/voice/agent/start",
                json={"avatarId": AVATAR, "channel": CHANNEL, "userUid": UID}).json()
    agent_id = agent["agentId"]
    print(f"Avatar started — speak into your mic. Ctrl+C to end.")

    svc_cfg       = AgoraServiceConfig()
    svc_cfg.appid = creds["appId"]
    svc           = AgoraService()
    svc.initialize(svc_cfg)

    conn_cfg = RTCConnConfig(
        client_role_type=ClientRoleType.CLIENT_ROLE_BROADCASTER,
        channel_profile=ChannelProfileType.CHANNEL_PROFILE_LIVE_BROADCASTING,
        auto_subscribe_audio=1,
        auto_subscribe_video=0,
        audio_recv_media_packet=0,
        audio_subs_options=AudioSubscriptionOptions(
            packet_only=0, pcm_data_only=1,
            bytes_per_sample=2, number_of_channels=CH, sample_rate_hz=RATE,
        ),
    )
    pub_cfg = RtcConnectionPublishConfig(
        audio_profile=AudioProfileType.AUDIO_PROFILE_DEFAULT,
        audio_scenario=AudioScenarioType.AUDIO_SCENARIO_AI_SERVER,
        is_publish_audio=True, is_publish_video=False,
        audio_publish_type=AudioPublishType.AUDIO_PUBLISH_TYPE_PCM,
        video_publish_type=VideoPublishType.VIDEO_PUBLISH_TYPE_NONE,
    )
    conn = svc.create_rtc_connection(conn_cfg, pub_cfg)
    conn.connect(creds["token"], CHANNEL, str(UID))

    local_user = conn.get_local_user()
    local_user.set_playback_audio_frame_before_mixing_parameters(CH, RATE)
    conn.register_audio_frame_observer(SpeakerObserver(), 0, None)
    conn.publish_audio()

    mic  = pa.open(format=pyaudio.paInt16, channels=CH, rate=RATE,
                   input=True, frames_per_buffer=CHUNK)
    loop = asyncio.get_event_loop()
    try:
        while True:
            pcm = await loop.run_in_executor(None, mic.read, CHUNK)
            conn.push_audio_pcm_data(pcm, RATE, CH)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        mic.stop_stream(); mic.close()
        requests.post(f"{BASE}/api/voice/agent/stop", json={"agentId": agent_id})
        conn.disconnect(); conn.release()
        svc.release()
        speaker.stop_stream(); speaker.close()
        pa.terminate()
        print("Session ended.")

asyncio.run(main())`}</pre>
          </div>
        </details>
      )}
    </div>
  );
}
