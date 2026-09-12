# Orchestrator — progresso (AvatarGarden repo: orchestrator/)

## Status: Fases 1-4 VALIDADAS (12/Sep) — E2E PASS

| Fase | Entrega | Validação |
|---|---|---|
| 1 | `g722.py` — encoder G722 em Python puro (port do SpanDSP/Asterisk) | BYTE-EXACT vs ffmpeg; 2.13 ms/frame de 20 ms |
| 2 | `asr_deepgram.py` — DeepgramStreamer (nova-3, language=multi, endpointing=500 via speech_final; utterance_end_ms=1000 fallback — 500 rejeitado pela API) | E2E: Cartesia TTS -> stream -> utterance 0.71-0.94s após fim da fala |
| 3 | `session.py` — loop completo: Agora (uid 777, PoC path) -> uplink -> Deepgram -> `POST localhost /api/voice/chat-completions?avatar=X&streaming=1` (mesmo cérebro: RAG/sensores/species/weather) -> SSE por frase |
| 4 | TTS Cartesia sonic-3 (.ai + header Cartesia-Version) -> G722 (fase 1) -> push_audio_encoded_data paced 20ms |

E2E (test_orchestrator_e2e.py — bot joga o usuário, sem humano):
- Canal Agora real; orchestrator entrou como uid 777 (POC path validado de novo)
- Bot publicou pergunta TTS em tempo real + silêncio streamado
- ASR capturou: "você pode me contar uma história curta sobre o rio marumbi"
- LLM localhost (marumbi, gpt-oss-120b) respondeu 791 chars em 9.94s (stream)
- Bot recebeu 1187 frames = 23.7s de áudio; ASR do reply confirma conteúdo
- Half-duplex: uplink descartado durante downlink (gate próprio, sem AIVAD)
- Auto-idle nativo (cost-guard resolvido na raiz)

## Fase 5 (próxima): roteamento por avatar
- `voiceBackend: "convoai" | "vps"` per-avatar; choke point = /api/voice/agent/start
  + /stop (device e web chamam os mesmos endpoints — firmware inalterado)
- Toggle em LLM Config → Voice; default convoai (fallback intacto)
- Web: testar decode G.722 no Web SDK; fallback = Opus p/ web + G722 p/ device

## Bugs/learnings desta fase
- Token Agora é uid-bound: o token do orquestrador DEVE ser mintado para
  uid 777 (o PoC fazia certo; orchestrator_main v1 usou uid 1 — join falhava
  silenciosamente, sem on_connected)
- `push_audio_pcm_data` exige buffer WRITABLE (bytearray, não bytes)
- Endpointing conta silêncio no áudio RECEBIDO: silêncio final precisa ser
  STREAMADO (áudio contínuo de mic sempre contém — resolvido no fluxo real)
- Pausas naturais ("Oi! ") disparam endpointing no meio da fala — tuning de
  endpointing/turn-detection fica para a fase 5+ (por orelha)
- python stdout em nohup: usar `python -u` (buffering escondeu logs inteiros)
- Latência percebida (utterance->1º frame): ~11-12s — mesmo perfil do ConvoAI
  (TTFT GWDG domina), como previsto no doc