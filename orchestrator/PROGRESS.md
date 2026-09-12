# Orchestrator — progresso (AvatarGarden repo: orchestrator/)

## Status: Fases 1-5 IMPLEMENTADAS E VALIDADAS (12/Sep) — Em produção

| Fase | Entrega | Validação |
|---|---|---|
| 1 | `g722.py` — encoder G722 em Python puro (port do SpanDSP/Asterisk) | BYTE-EXACT vs ffmpeg; 2.13 ms/frame de 20 ms |
| 2 | `asr_deepgram.py` — DeepgramStreamer (nova-3, language=multi, endpointing=500 via speech_final; utterance_end_ms=1000 fallback — 500 rejeitado pela API) | E2E: Cartesia TTS -> stream -> utterance 0.71-0.94s após fim da fala |
| 3 | `session.py` — loop completo: Agora (uid 777, PoC path) -> uplink -> Deepgram -> `POST localhost /api/voice/chat-completions?avatar=X&streaming=1` (mesmo cérebro: RAG/sensores/species/weather) -> SSE por frase |
| 4 | TTS Cartesia sonic-3 (.ai + header Cartesia-Version) -> G722 (fase 1) -> push_audio_encoded_data paced 20ms |
| 5 | Roteamento por avatar + plumbing na plataforma: `voiceBackend: convoai | vps` em llmDefaults (survive admin-defaults via save/clear), choke point = /api/voice/agent/start + /stop, spawn subprocess (orchestrator_main) com adopt por canal, stop via SIGTERM, toggle "Voice engine" em LLM Config -> Voice, voice page esconde toggle streaming p/ vps (streaming nativo) | spawn/adopt/stop via API; E2E bot; usuário validou web + device por áudio |

E2E (test_orchestrator_e2e.py — bot joga o usuário, sem humano):
- Canal Agora real; orchestrator entrou como uid 777 (POC path validado de novo)
- Bot publicou pergunta TTS em tempo real + silêncio streamado
- ASR capturou: "você pode me contar uma história curta sobre o rio marumbi"
- LLM localhost (marumbi, gpt-oss-120b) respondeu 791 chars em 9.94s (stream)
- Bot recebeu 1187 frames = 23.7s de áudio; ASR do reply confirma conteúdo
- Latência percebida (utterance->1º frame): ~11-12s — mesmo perfil do ConvoAI
  (TTFT GWDG domina), como previsto no doc

## Áudio por cliente (12/Sep, pós-feedback de áudio)
- **Web -> Opus** (opuslib_next, 1 pacote variável por frame de 20ms,
  EncodedAudioFrameInfo 16kHz/320 samples, full-duplex + barge-in — browser AEC)
- **Device -> G722** (half-duplex; mic reabre quando o PLAYBACK termina
  — _push_busy tracking — não quando o pipeline inteiro drena)
- Barge-in: contador de geração — publisher corta o áudio antigo a 20ms de
  granularidade; turnos interrompidos não sujam o histórico
- Instrumentação de sessão: "first LLM sentence ready", "tts latency" por frase
- Reprodução drift-free: agendamento ABSOLUTO de 20ms (era sleep-após-push,
  drift acumulava -> voz lenta/jitter)

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
- Opus: pacotes são de comprimento VARIÁVEL — concatenar e re-fatiar o stream
  corrompe as fronteiras de pacote (ASR do reply voltou vazio; fix: 1 pacote
  por frame de 20ms, push individual)
- Limpeza de código pode comer variáveis (frame_info sumiu no refactor —
  publisher morria com NameError e a thread SDK morria junto, downlink mudo);
  publisher agora captura exceções
- pkill -f casa o próprio shell — usar padrão "[o]rchestrator_main"
- Admin Defaults CLEAR reseta voiceBackend para convoai (semântica escolhida:
  save/clear manda; switch é instant-apply + registrado no save)

## Custo (modo vps por avatar)
- ConvoAI $0.10/min: ZERADO para sessões vps
- Agora RTC: 2 uids (user + 777) -> ~2 min RTC por min de conversa; 10k
  min/mês free; metros contam PRESENÇA (silêncio conta — auto-idle do
  orquestrador desconecta em 180s; eco-mode de firmware seria a fase 6)
- Deepgram ~$0.005/min de fala + Cartesia ~1 credit/char (igual antes)

## Próximos passos
1. **Fase 6 (visão, pendente)**: RTC self-hosted — UDP/RTP G722 direto
   escultura<->VPS (~20-50ms, $0 presença 24/7), toggle de transporte sob o
   mesmo seletor do voiceBackend (matriz backend x transporte: convoai só
   agora; vps/um790 x agora/rtp); firmware: módulo rtc_udp (~300-500 linhas)
   ao lado do rtc_proc — pipeline de áudio inalterado
2. **Tuning device (por orelha, próxima sessão na bancada)**: verificar
   qualidade/tx do uplink do device no orquestrador (PoC só validou frames
   chegando; nunca a qualidade — suspeita de bumps: resample ausente ou
   pacing); adiciona resampler/jitter-buffer no uplink se necessário;
   medir lag do frame pump (instrumentação já existe)
3. Web: já smooth pós-Opus; monitorar
4. WhatsApp (marumbi-02): nova interface para o mesmo pipeline
   (endpoint-agnostic)
5. Tuning fino: endpointing 500ms pode cortar pausas naturais; testar 650ms
   p/ device; modelo voiceChat qwen3-30b-a3b como default rápido p/ marumbi
   (TTFT menor) — avaliar por orelha
