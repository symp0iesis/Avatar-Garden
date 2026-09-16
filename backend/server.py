import asyncio
import io
import json
import os
import re
import socket
import threading
import signal
import subprocess
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()
print("Loaded from .env file.")

from flask import Flask, Response, jsonify, make_response, request, send_file
from flask_cors import CORS
from llama_index.core.tools.query_engine import QueryEngineTool
from utils.llm_tooling import LLM
from utils.avatar_setup import (
    avatar_llms,
    avatar_rag_tools,
    avatar_sensor_tools,
    avatars_path,
    generate_avatars_config,
    sensor_config_from_avatar,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_CHAT_MODEL,
    DEFAULT_TEXT_QUERY_PROVIDER,
    DEFAULT_TEXT_QUERY_MODEL,
    DEFAULT_SENSOR_PROVIDER,
    DEFAULT_SENSOR_MODEL,
    SENSOR_SYSTEM_PROMPT,
    TEXT_QUERY_SYSTEM_PROMPT,
)
from utils.processing_pipelines import OpenAIRealtimeClient
from utils.utils import (
    RAG,
    build_or_load_index,
    detect_web_search_intent,
    extract_keywords_multilingual,
    fetch_system_prompt_from_gdoc,
    fetch_text_index_query,
    generate_context_aware_keywords_for_multilingual_text_index_search,
    format_history_as_string,
    pcm_to_wav_bytes,
    prepare_query_engines,
    transcribe_audio,
    SensorsTool,
    web_search,
)
from werkzeug.utils import secure_filename
import requests

# === Initialize Flask ===
app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    from werkzeug.exceptions import HTTPException
    code = e.code if isinstance(e, HTTPException) else 500
    return jsonify({"error": str(e), "traceback": traceback.format_exc()}), code

UPLOAD_DIR = "data/uploaded_experiences"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DEBUG_LOG_DIR = "debug"
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)

# Timestamped snapshots of avatars.json, taken before every write. Recovery
# path for accidental default wipes (ISSUES.md 13): copy a snapshot back over
# avatars.json and restart.
AVATARS_HISTORY_DIR = "avatars_history"
AVATARS_HISTORY_KEEP = 100
os.makedirs(AVATARS_HISTORY_DIR, exist_ok=True)

def save_avatars(avatars, reason=""):
    """Single write path for avatars.json: snapshot the current state, then write."""
    import shutil
    if os.path.exists(avatars_path):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        shutil.copy2(avatars_path, os.path.join(AVATARS_HISTORY_DIR, f"avatars-{ts}.json"))
        snapshots = sorted(os.listdir(AVATARS_HISTORY_DIR))
        for old in snapshots[:-AVATARS_HISTORY_KEEP]:
            os.remove(os.path.join(AVATARS_HISTORY_DIR, old))
    json.dump(avatars, open(avatars_path, "w"))
    print(f"[avatars] saved ({reason or 'unspecified'}) — snapshot in {AVATARS_HISTORY_DIR}/")

def debug_log(avatar_id, section, content):
    """Append a timestamped debug entry to debug/<avatar_id>.log."""
    from datetime import datetime as _dt
    path = os.path.join(DEBUG_LOG_DIR, f"{avatar_id}.log")
    ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 60
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n{separator}\n[{ts}] [{section}]\n{separator}\n{content}\n")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GWDG_API_KEY = os.getenv("GWDG_API_KEY")
GWDG_API_BASE = os.getenv("GWDG_API_BASE")
BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", "")
LLM_PROVIDERS_PATH = "llm_providers.json"
INTEGRATIONS_PATH = "integrations.json"

INTEGRATION_KEYS = [
    "BRAVE_SEARCH_API_KEY",
    "AGORA_APP_ID",
    "AGORA_APP_CERTIFICATE",
    "AGORA_CUSTOMER_ID",
    "AGORA_CUSTOMER_SECRET",
    "DEEPGRAM_API_KEY",
    "CARTESIA_API_KEY",
]

def _load_integrations():
    if os.path.exists(INTEGRATIONS_PATH):
        try:
            return json.load(open(INTEGRATIONS_PATH))
        except Exception:
            pass
    return {}

def _save_integrations(data):
    json.dump(data, open(INTEGRATIONS_PATH, "w"), indent=2)

def get_integration_key(name):
    """Read an integration key: integrations.json first, then os.getenv() fallback."""
    val = _load_integrations().get(name, "")
    if val:
        return val
    return os.getenv(name, "")

if not os.path.exists(INTEGRATIONS_PATH):
    _save_integrations({k: "" for k in INTEGRATION_KEYS})
HEALTH_CACHE_PATH = "llm_health_cache.json"
_models_last_refreshed = None  # ISO timestamp, set after each model list refresh
_model_health_cache = json.load(open(HEALTH_CACHE_PATH)) if os.path.exists(HEALTH_CACHE_PATH) else {}

_sensor_cache = {}  # avatar_id -> {"ts": float, "summary": str}
SENSOR_CACHE_TTL = 120  # seconds

def _fetch_sensor_summary(avatar_id):
    """Fetch latest sensor readings and return a formatted text summary. Cached per avatar."""
    import time
    cached = _sensor_cache.get(avatar_id)
    if cached and (time.time() - cached["ts"]) < SENSOR_CACHE_TTL:
        return cached["summary"]

    sensor_config = avatar_sensor_tools.get(avatar_id)
    if not sensor_config:
        return None
    sensor_url = sensor_config.get("sensor_url", "")
    if not sensor_url:
        return None

    # Fetch last 10 readings
    fetch_url = sensor_url if "results=" in sensor_url else sensor_url + ("&" if "?" in sensor_url else "?") + "results=10"
    try:
        resp = requests.get(fetch_url, headers={"Accept": "application/json", "User-Agent": "AvatarGarden/1.0"}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[sensor cache] fetch failed for avatar {avatar_id}: {e}")
        return None

    # Parse field names and most recent non-null values
    lines = []
    if "channel" in data and "feeds" in data:
        channel = data["channel"]
        field_map = {f"field{i}": channel[f"field{i}"] for i in range(1, 9) if f"field{i}" in channel}
        feeds = data["feeds"]
        # Most recent reading with at least one non-null value
        latest = next((f for f in reversed(feeds) if any(f.get(k) is not None for k in field_map)), None)
        if latest:
            lines.append(f"Latest reading ({latest.get('created_at', 'unknown time')}):")
            for field_key, field_name in field_map.items():
                val = latest.get(field_key)
                if val is not None:
                    lines.append(f"  {field_name}: {val}")
    elif "feeds" in data:
        feeds = data["feeds"]
        latest = feeds[-1] if feeds else None
        if latest:
            ts = latest.get("timestamp") or latest.get("created_at", "unknown time")
            lines.append(f"Latest reading ({ts}):")
            for k, v in latest.items():
                if k not in ("timestamp", "created_at", "entry_id") and v is not None:
                    lines.append(f"  {k}: {v}")

    if not lines:
        return None

    summary = "\n".join(lines)
    _sensor_cache[avatar_id] = {"ts": time.time(), "summary": summary}
    print(f"[sensor cache] refreshed for avatar {avatar_id}")
    return summary


def _persist_health_cache():
    try:
        json.dump(_model_health_cache, open(HEALTH_CACHE_PATH, "w"))
    except Exception as e:
        print(f"[health cache] failed to persist: {e}")

if not os.path.exists(LLM_PROVIDERS_PATH):
    json.dump([
        {"id": "gwdg", "name": "JLU", "provider_key": "gwdg",
         "api_base": "https://api.hrz.uni-giessen.de", "api_key_env": "GWDG_API_KEY", "models": []},
        {"id": "openrouter", "name": "OpenRouter", "provider_key": "openai",
         "api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "models": []},
        {"id": "ollama", "name": "Ollama Cloud", "provider_key": "ollama",
         "api_base": "https://ollama.com", "api_key_env": "OLLAMA_API_KEY", "models": []},
    ], open(LLM_PROVIDERS_PATH, "w"), indent=2)


# ─── Shared helpers (used by /api/chat and /api/voice/chat-completions) ───

# Marker that starts the sensor/function-calling section of avatar system prompts.
# Voice/realtime routes truncate the prompt at this marker (tools are handled separately).
SENSOR_SECTION_MARKER = "You also have access to sensory data"

# Injected into the system prompt alongside the always-fetch sensor snapshot,
# in both the chat and voice pipelines.
SENSOR_SNAPSHOT_INSTRUCTION = (
    "\n\nCURRENT SENSOR SNAPSHOT:\n{summary}\n"
    "Use this for simple current-reading questions. Call analyze_sensor_data() "
    "only for trend analysis or questions requiring historical data."
)

# Emoji / pictograph ranges commonly emitted by chat models.
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")

def sanitize_for_tts(text):
    """
    Strip formatting that TTS vocalizes. Verified 2026-08-11: Cartesia (sonic-2
    AND sonic-3) literally speaks '|' as "vertical bar"; models also emit
    *markdown* and emojis into voice replies despite prompt instructions.
    """
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)   # [text](url) -> text
    t = re.sub(r"^\s*[-•]\s+", "", t, flags=re.M)        # bullet markers
    t = re.sub(r"[*_`#|]+", " ", t)                      # markdown chars + pipes
    t = _EMOJI_RE.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

# Quantities a live sensor feed can answer (EN/DE/PT). A web-search flag whose
# query matches one of these is suppressed for avatars that have sensors — small
# keyword-gen LLMs misjudge this boundary (tested: 8B fails 0-1/5 on sensor topics
# even with few-shot examples), so it is enforced deterministically here.
SENSOR_TOPIC_TERMS = [
    "temperature", "temperatur", "temperatura",
    "water quality", "wasserqualität", "qualidade da água",
    "ph", "humidity", "luftfeuchtigkeit", "umidade",
    "conductivity", "leitfähigkeit", "condutividade",
    "oxygen", "sauerstoff", "oxigênio",
    "co2", "tds", "pressure", "druck", "pressão",
    "sensor", "reading", "messwert",
]

def suppress_sensor_web_search(web_query, avatar_id):
    """Drop a web-search flag if the avatar has sensors and the query is sensor-covered."""
    if not web_query or not avatar_sensor_tools.get(avatar_id):
        return web_query
    q = web_query.lower()
    hit = next((t for t in SENSOR_TOPIC_TERMS if re.search(rf"\b{re.escape(t)}\b", q)), None)
    if hit:
        print(f"[web search] Suppressed {web_query!r} — sensor-covered topic ({hit!r})")
        return None
    return web_query

MODEL_FAMILIES = ['gemma', 'qwen', 'llama', 'mistral', 'deepseek', 'codestral',
                  'phi', 'falcon', 'mixtral', 'command', 'claude', 'gpt',
                  'devstral', 'medgemma', 'glm', 'kimi', 'minimax']

def _extract_family(model_name):
    if not model_name:
        return None
    lower = model_name.lower()
    return next((f for f in MODEL_FAMILIES if f in lower), None)

def _is_free_tier(model_name):
    return model_name.endswith(':free') or ':free:' in model_name

def fallback_model(provider_id, model_name):
    """
    When an admin-default model is offline, find the best available substitute.
    Tier 1: same provider, same family, online.
    Tier 2: any provider, same family, online.
    Tier 3: any online model on any provider.
    Returns (provider_id, model_name) — unchanged if the model is healthy or cache is empty.
    """
    if not _model_health_cache:
        return provider_id, model_name
    if _model_health_cache.get(model_name) != 'offline':
        return provider_id, model_name

    try:
        providers = json.load(open(LLM_PROVIDERS_PATH, "r"))
    except Exception:
        return provider_id, model_name

    visible = [p for p in providers if not p.get('hidden')]
    family = _extract_family(model_name)

    def _candidate(m):
        return _model_health_cache.get(m) == 'online' and not _is_free_tier(m)

    # Tier 1: same provider, same family
    same_provider = next((p for p in visible if p['id'] == provider_id), None)
    if same_provider and family:
        t1 = next((m for m in same_provider.get('models', [])
                   if _extract_family(m) == family and _candidate(m)), None)
        if t1:
            print(f"[fallback] {model_name} offline → tier-1 fallback: {t1} ({provider_id})")
            return provider_id, t1

    # Tier 2: any provider, same family
    if family:
        for p in visible:
            t2 = next((m for m in p.get('models', [])
                       if _extract_family(m) == family and _candidate(m)), None)
            if t2:
                print(f"[fallback] {model_name} offline → tier-2 fallback: {t2} ({p['id']})")
                return p['id'], t2

    # Tier 3: any online non-free model
    for p in visible:
        t3 = next((m for m in p.get('models', []) if _candidate(m)), None)
        if t3:
            print(f"[fallback] {model_name} offline → tier-3 fallback: {t3} ({p['id']})")
            return p['id'], t3

    print(f"[fallback] {model_name} offline — no online fallback found, proceeding anyway")
    return provider_id, model_name

def create_llm_instance(provider_id, model, system_prompt, temperature=0.7,
                        top_k=40, top_p=1.0, task_name="unknown", providers=None):
    """Create an LLM instance for any task. Single source of truth for LLM construction."""
    if providers is None:
        providers = json.load(open(LLM_PROVIDERS_PATH, "r"))

    provider = next((p for p in providers if p["id"] == provider_id), None)
    if not provider:
        raise ValueError(f"Unknown LLM provider: {provider_id}")

    provider_key = provider["provider_key"]
    api_key_env = provider.get("api_key_env", "")
    api_base = provider.get("api_base", "")

    api_key = provider.get("api_key", "") or (os.getenv(api_key_env, "") if api_key_env else "")
    if not api_key:
        if provider_key == "openai":
            api_key = OPENAI_API_KEY
        elif provider_key == "gwdg":
            api_key = GWDG_API_KEY
        elif provider_key == "ollama":
            api_key = os.getenv("OLLAMA_API_KEY", "")
    if not api_base and provider_key == "gwdg" and GWDG_API_BASE:
        api_base = GWDG_API_BASE

    print(f"Creating LLM for {task_name}: provider={provider_id}, model={model}, "
          f"temp={temperature}, top_k={top_k}, top_p={top_p}")

    if provider_key == "openai":
        llm_params = {
            "provider": "openai", "openai_model": model, "system_prompt": system_prompt,
            "openai_api_key": api_key, "temperature": temperature, "top_k": top_k, "top_p": top_p,
        }
        if api_base:
            llm_params["openai_api_base"] = api_base
    elif provider_key == "ollama":
        llm_params = {
            "provider": "ollama", "ollama_model": model, "system_prompt": system_prompt,
            "temperature": temperature, "top_k": top_k, "top_p": top_p,
        }
        if api_key:
            llm_params["ollama_api_key"] = api_key
        if api_base:
            llm_params["ollama_api_base"] = api_base
    else:
        llm_params = {
            "provider": "gwdg", "gwdg_model": model, "system_prompt": system_prompt,
            "temperature": temperature, "top_k": top_k, "top_p": top_p,
        }
        if api_key:
            llm_params["gwdg_api_key"] = api_key
        if api_base:
            llm_params["gwdg_api_base"] = api_base

    return LLM(**llm_params)


def resolve_llm_defaults(avatar_id, user_params=None):
    """
    Resolve LLM parameters via cascade: user request → admin defaults → global defaults.
    Returns (resolved_dict, sources_dict).
    """
    avatars = json.load(open(avatars_path, "r"))
    avatar_obj = next((a for a in avatars if a["id"] == avatar_id), None)
    admin_defaults = avatar_obj.get("llmDefaults", {}) if avatar_obj else {}

    chat_defaults = admin_defaults.get("chat", {})
    text_query_defaults = admin_defaults.get("textQuery", {})
    sensor_defaults = admin_defaults.get("sensor", {})

    if user_params is None:
        user_params = {}
    sources = {}

    def _resolve(req_val, default_val, global_val, key):
        if req_val is not None:
            if default_val is not None and default_val == req_val:
                sources[key] = "Admin defaults"
            else:
                sources[key] = "user"
            return req_val
        elif default_val is not None:
            sources[key] = "Admin defaults"
            return default_val
        else:
            sources[key] = "global defaults"
            return global_val

    resolved = {}
    resolved["chat_provider"] = _resolve(
        user_params.get("chatProvider"), chat_defaults.get("provider"), DEFAULT_CHAT_PROVIDER, "chat_provider")
    resolved["chat_model"] = _resolve(
        user_params.get("chatModel"), chat_defaults.get("model"), DEFAULT_CHAT_MODEL, "chat_model")
    resolved["temperature"] = _resolve(
        user_params.get("temperature"), chat_defaults.get("temperature"), 0.7, "temperature")
    resolved["top_k"] = _resolve(
        user_params.get("topK"), chat_defaults.get("top_k"), 40, "top_k")
    resolved["top_p"] = _resolve(
        user_params.get("topP"), chat_defaults.get("top_p"), 1.0, "top_p")

    # Text query — falls back to chat provider if unset
    resolved["text_query_provider"] = _resolve(
        user_params.get("textQueryProvider"), text_query_defaults.get("provider"), resolved["chat_provider"], "text_query_provider")
    if sources["text_query_provider"] == "global defaults":
        sources["text_query_provider"] = "inherited from chat"
    resolved["text_query_model"] = _resolve(
        user_params.get("textQueryModel"), text_query_defaults.get("model"), DEFAULT_TEXT_QUERY_MODEL, "text_query_model")

    # Sensor — falls back to chat provider if unset
    resolved["sensor_provider"] = _resolve(
        user_params.get("sensorProvider"), sensor_defaults.get("provider"), resolved["chat_provider"], "sensor_provider")
    if sources["sensor_provider"] == "global defaults":
        sources["sensor_provider"] = "inherited from chat"
    resolved["sensor_model"] = _resolve(
        user_params.get("sensorModel"), sensor_defaults.get("model"), DEFAULT_SENSOR_MODEL, "sensor_model")

    # Voice chat — response model for voice sessions, separate from text chat
    # because voice is latency-critical (the reply blocks TTS). Falls back to
    # the resolved text-chat model when unset.
    voice_defaults = admin_defaults.get("voiceChat", {})
    resolved["voice_chat_provider"] = voice_defaults.get("provider") or resolved["chat_provider"]
    resolved["voice_chat_model"] = voice_defaults.get("model") or resolved["chat_model"]
    sources["voice_chat_model"] = "Admin defaults" if voice_defaults.get("model") else "inherited from chat"

    return resolved, sources


def _avatar_keyword_mode(avatar_id):
    """Per-avatar keyword mode: 'llm' (cloud GWDG, context-aware, default) or
    'vps'/'local' (VPS keyword extraction — spaCy on this server, fast)."""
    try:
        avatars_cfg = json.load(open(avatars_path, "r"))
        cfg = next((a for a in avatars_cfg if str(a.get("id")) == str(avatar_id)), None)
        mode = str((cfg or {}).get("keywordMode", "llm")).lower()
        return "local" if mode in ("local", "vps") else "llm"
    except Exception:
        return "llm"


def inject_rag_context(avatar_id, chat_history, conversation_for_rag, text_query_llm,
                       verbose=True, web_search_enabled=True):
    """
    Inject RAG context into the last message of chat_history (in-place).
    verbose=True uses the full chat-style wrapper; False uses a compact voice wrapper.
    Returns (rag_injected: bool, web_search_query: str | None, timings: dict).
    The web_search_query is detected in the same LLM call as keyword generation (RAG path),
    or via a dedicated lightweight call (no-RAG path).
    """
    timings = {}
    tools = avatar_rag_tools.get(avatar_id)
    has_rag = tools and tools[0] is not None
    has_sensor = bool(avatar_sensor_tools.get(avatar_id))

    if not has_rag:
        if not web_search_enabled:
            return False, None, timings
        # No RAG — run a cheap dedicated web search intent check
        _t = time.perf_counter()
        web_search_query = detect_web_search_intent(text_query_llm, conversation_for_rag, has_sensor=has_sensor)
        timings["keyword_gen_ms"] = round((time.perf_counter() - _t) * 1000)
        return False, web_search_query, timings

    rag_languages = tools[4] if len(tools) > 4 else ['en', 'de']

    _t = time.perf_counter()
    keyword_mode = _avatar_keyword_mode(avatar_id)
    if keyword_mode == "local":
        # VPS keyword extraction: spaCy POS extraction on the last user message, no LLM
        # round trip (~50ms vs ~1s GWDG). Tradeoff: no conversation-aware context
        # resolution for follow-ups. Best for monolingual corpus + interactions.
        try:
            last_user = next(
                (m.get("text", "") for m in reversed(conversation_for_rag)
                 if str(m.get("sender", "")).lower() == "user"),
                "",
            )
            keywords_by_lang = extract_keywords_multilingual(last_user, rag_languages) if last_user.strip() else {}
            web_search_query = None
            if not any(keywords_by_lang.values()):
                raise ValueError("VPS keyword extraction produced no keywords")
            print(f"[keywords] VPS keyword extraction for avatar {avatar_id}")
        except Exception as e:
            print(f"[keywords] VPS extraction failed ({e}) — falling back to LLM mode")
            keywords_by_lang, web_search_query = generate_context_aware_keywords_for_multilingual_text_index_search(
                text_query_llm, conversation_for_rag, rag_languages, has_sensor=has_sensor
            )
    else:
        keywords_by_lang, web_search_query = generate_context_aware_keywords_for_multilingual_text_index_search(
            text_query_llm, conversation_for_rag, rag_languages, has_sensor=has_sensor
        )
    timings["keyword_gen_ms"] = round((time.perf_counter() - _t) * 1000)

    _t = time.perf_counter()
    context = RAG(tools, None, keywords_by_lang=keywords_by_lang)
    timings["rag_retrieval_ms"] = round((time.perf_counter() - _t) * 1000)

    if verbose:
        wrapper = (
            " <End of User message>.        <<IMPORTANT: The following info is a RAG injection "
            "to provide you with helpful context. The user did not send this: Here is relevant "
            "information (Sometimes the text-retrieval has relevant information that the "
            "vector-retrieval doesn't, or vice versa. Look through each comprehensively, to "
            "extract the information you need. Even if the Vector-retrieval says there's no "
            "information available, still scrutinize the Text-retrieval results to fetch relevant "
            "info (What language was the user's last message in? Make sure to respond in the same "
            "language.) This instruction is in English, but your response should be in whatever "
            "language the user messaged you in:>>  "
        )
    else:
        wrapper = " <End of User message>. <<Context from knowledge base: "

    chat_history[-1]["content"] += wrapper + context + (">>" if not verbose else "")
    if not web_search_enabled:
        return True, None, timings

    return True, web_search_query, timings


def handle_sensor_tool_call(response_text, avatar_id, sensor_provider, sensor_model,
                            llm, chat_history, providers=None):
    """
    If response_text contains analyze_sensor_data(), run the sensor tool and re-invoke LLM.
    Returns (final_response, was_handled).
    """
    sensor_config = avatar_sensor_tools.get(avatar_id)
    if not sensor_config or "analyze_sensor_data" not in response_text:
        return response_text, False

    q_start = response_text.find('user_query="') + 12
    q_end = response_text.find('")', q_start)
    query = response_text[q_start:q_end]
    if not query:
        return response_text, False

    sensor_llm = create_llm_instance(
        sensor_provider, sensor_model, SENSOR_SYSTEM_PROMPT,
        task_name="sensor", providers=providers,
    )
    sensor_tool = SensorsTool(
        sensor_llm,
        sensor_url=sensor_config["sensor_url"],
        sensor_description=sensor_config.get("sensor_description"),
    )
    analysis = str(sensor_tool(query))
    print(f"Sensor analysis: {analysis}")

    results_msg = (
        "\nHere is the output of analyze_sensor_data(): " + analysis
        + " Respond to the user accordingly. Do not provide any subjective Lahn-specific "
        "evaluation of this data, just focus on the quantitative result. And do not return "
        "a function call. What language was the user's last message in? Make sure to respond "
        "in the same language."
    )
    chat_history.append({"role": "assistant", "content": results_msg})
    final = llm.complete(chat_history).text

    # Guard against recursive function calls
    if "analyze_sensor_data" in final:
        final = analysis

    return final.replace("*", ""), True


VOICE_WEB_SEARCH_SCHEMA = """
FUNCTION:
You have access to this function. Call it ONLY when the user asks about recent
events, news, projects, or information that requires live internet data (not
covered by your knowledge base or sensor feed).

web_search(query="your search query")
   - Your query MUST include location/context terms that identify your
     bioregion (river name, region, municipality).
   - Do NOT call for questions about ecology, fauna, history, or community
     stories — those are already covered by the knowledge base context above.
   - Output ONLY the function call line, nothing else.
"""


def handle_web_search_tool_call(response_text, avatar_id, llm, chat_history):
    """
    If response_text contains web_search(query=...), run Brave and re-invoke the
    LLM with results labeled as an external source. Returns (final_response,
    was_handled). Mirrors handle_sensor_tool_call; used by the voice callback
    (buffered + streaming). The per-avatar webSearchEnabled flag gates injection.
    """
    if 'web_search(query="' not in response_text:
        return response_text, False

    q_start = response_text.find('web_search(query="') + len('web_search(query="')
    q_end = response_text.find('")', q_start)
    query = response_text[q_start:q_end] if q_end > q_start else ""
    if not query:
        return response_text, False

    print(f"[web search] Voice tool-invoked query: {query!r}")
    try:
        ws_results = web_search(query, count=5, api_key=get_integration_key("BRAVE_SEARCH_API_KEY"))
    except Exception as e:
        print(f"[Voice] Web search failed (continuing with original response): {e}")
        return response_text, False
    debug_log(avatar_id, "VOICE/WEB_SEARCH_RESULTS", ws_results)

    chat_history.append({"role": "assistant", "content": response_text})
    chat_history.append({"role": "user", "content":
        f"WEB SEARCH RESULTS (live internet data retrieved for you — external source, "
        f"NOT from your knowledge base):\n{ws_results}\n"
        "Use these results where they are relevant to the user's question; ignore "
        "anything that does not apply to your bioregion. Respond conversationally in "
        "the user's language. Do not return a function call."
    })
    final = llm.complete(chat_history).text

    # Guard against recursive function calls
    if 'web_search(query="' in final:
        final = ws_results

    return final.replace("*", ""), True


def _build_model_checks(provider_ids_filter=None):
    """Build list of model checks, optionally filtered by provider IDs."""
    providers = json.load(open(LLM_PROVIDERS_PATH, "r"))

    model_checks = []
    for provider in providers:
        provider_key = provider["provider_key"]

        # Skip hidden providers
        if provider.get("hidden"):
            continue

        # Filter by provider id (unique per provider, unlike provider_key)
        if provider_ids_filter and provider["id"] not in provider_ids_filter:
            continue

        api_key_env = provider.get("api_key_env", "")
        api_base = provider.get("api_base", "")

        # Load API key: provider JSON first, then env var fallback
        api_key = provider.get("api_key", "") or (os.getenv(api_key_env, "") if api_key_env else "")

        # Special handling for legacy env vars
        if not api_key:
            if provider_key == "openai":
                api_key = OPENAI_API_KEY
            elif provider_key == "gwdg":
                api_key = GWDG_API_KEY
            elif provider_key == "ollama":
                api_key = os.getenv("OLLAMA_API_KEY", "")

        if not api_base:
            if provider_key == "gwdg" and GWDG_API_BASE:
                api_base = GWDG_API_BASE

        health_check_timeout = 15

        for model_name in provider.get("models", []):
            model_checks.append({
                "model_name": model_name,
                "provider_key": provider_key,
                "api_key": api_key,
                "api_base": api_base,
                "timeout": health_check_timeout
            })

    return model_checks


def _health_check_reason(e):
    """Return a short human-readable reason for a failed health check."""
    msg = str(e)
    if "429" in msg:
        return "rate limited (429)"
    if "404" in msg:
        return "not a chat model (404)"
    if "400" in msg:
        return "unavailable on provider's end (400)"
    if "500" in msg:
        return "provider server error (500)"
    if isinstance(e, requests.exceptions.Timeout) or "timed out" in msg.lower() or "timeout" in msg.lower():
        return "timed out"
    return f"error ({msg[:60]})"


def _check_model(check_data):
    """Check a single model's health, with one retry on 429."""
    import time, random
    from utils.llm_tooling import LLM

    model_name = check_data["model_name"]
    provider_key = check_data["provider_key"]
    api_key = check_data["api_key"]
    api_base = check_data["api_base"]
    timeout = check_data["timeout"]

    def build_llm():
        llm_params = {
            'provider': provider_key,
            'system_prompt': "You are a helpful assistant.",
            'timeout': timeout
        }
        if provider_key == "openai":
            llm_params['openai_model'] = model_name
            llm_params['openai_api_key'] = api_key
            if api_base:
                llm_params['openai_api_base'] = api_base
        elif provider_key == "gwdg":
            llm_params['gwdg_model'] = model_name
            if api_key:
                llm_params['gwdg_api_key'] = api_key
            if api_base:
                llm_params['gwdg_api_base'] = api_base
        elif provider_key == "ollama":
            llm_params['ollama_model'] = model_name
            if api_key:
                llm_params['ollama_api_key'] = api_key
            if api_base:
                llm_params['ollama_api_base'] = api_base
        else:
            llm_params['provider'] = 'gwdg'
            llm_params['gwdg_model'] = model_name
            if api_key:
                llm_params['gwdg_api_key'] = api_key
            if api_base:
                llm_params['gwdg_api_base'] = api_base
        return LLM(**llm_params)

    MAX_RETRIES = 6
    for attempt in range(MAX_RETRIES + 1):
        try:
            build_llm().complete(prompt="ping")
            suffix = f" (after {attempt} retries)" if attempt > 0 else ""
            print(f"Health check: {model_name} ({provider_key}) -> online{suffix}")
            return (model_name, "online")
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES:
                # Rate limited — backoff with jitter and try again
                delay = (attempt + 1) * 2 + random.uniform(0, 3)
                time.sleep(delay)
                continue
            reason = _health_check_reason(e)
            print(f"Health check: {model_name} ({provider_key}) -> offline ({reason})")
            return (model_name, "offline")


def _run_health_checks(model_checks, max_workers):
    """Run health checks with specified parallelism."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    health_status = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(_check_model, check_data): check_data["model_name"]
            for check_data in model_checks
        }
        for future in as_completed(future_to_model):
            model_name, status = future.result()
            health_status[model_name] = status

    return health_status


@app.route("/api/health/llm/fast", methods=["GET"])
def llm_health_fast():
    """Health check for fast providers (OpenAI, GWDG) - high parallelism."""
    global _model_health_cache
    model_checks = _build_model_checks(provider_ids_filter=["openai", "gwdg"])
    results = _run_health_checks(model_checks, max_workers=10)
    _model_health_cache.update(results)
    _persist_health_cache()
    print(f"Fast health check results: {results}")
    return jsonify(results)


@app.route("/api/health/llm/slow", methods=["GET"])
def llm_health_slow():
    """Health check for slow/rate-limited providers (Ollama) - limited parallelism."""
    global _model_health_cache
    model_checks = _build_model_checks(provider_ids_filter=["ollama"])
    results = _run_health_checks(model_checks, max_workers=8)
    _model_health_cache.update(results)
    _persist_health_cache()
    print(f"Slow health check results: {results}")
    return jsonify(results)


def _check_openrouter_model(model_id, api_key):
    """Check OpenRouter model health via its endpoints metadata API (no inference needed)."""
    try:
        resp = requests.get(
            f"https://openrouter.ai/api/v1/models/{model_id}/endpoints",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        endpoints = resp.json().get("data", {}).get("endpoints", [])
        if not endpoints:
            return model_id, "offline"
        best_uptime = max((e.get("uptime_last_5m") or 0) for e in endpoints)
        status = "online" if best_uptime > 0 else "offline"
        return model_id, status
    except Exception as e:
        return model_id, "offline"


@app.route("/api/health/llm/openrouter", methods=["GET"])
def llm_health_openrouter():
    """Health check for OpenRouter using uptime metadata — no inference calls needed."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    providers = json.load(open(LLM_PROVIDERS_PATH, "r"))
    provider = next((p for p in providers if p["id"] == "openrouter"), None)
    if not provider:
        return jsonify({}), 200

    api_key = os.getenv(provider.get("api_key_env", ""), "")
    models = provider.get("models", [])

    results = {}
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_model = {
            executor.submit(_check_openrouter_model, m, api_key): m for m in models
        }
        for future in as_completed(future_to_model):
            model_id, status = future.result()
            results[model_id] = status

    global _model_health_cache
    _model_health_cache.update(results)
    _persist_health_cache()
    online = sum(1 for s in results.values() if s == "online")
    print(f"OpenRouter health check: {online}/{len(models)} online")
    return jsonify(results)


@app.route("/api/health/llm", methods=["GET"])
def llm_health():
    """
    Combined health check for all providers.
    Returns a map of model names to their status ('online' or 'offline').
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Build all model checks
    model_checks = _build_model_checks()

    # Separate by provider
    ollama_checks = [m for m in model_checks if m["provider_key"] == "ollama"]
    other_checks = [m for m in model_checks if m["provider_key"] != "ollama"]

    health_status = {}

    # Run both groups in parallel (but with different parallelism internally)
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both groups
        fast_future = executor.submit(_run_health_checks, other_checks, 10)
        slow_future = executor.submit(_run_health_checks, ollama_checks, 2)

        # Collect results
        health_status.update(fast_future.result())
        health_status.update(slow_future.result())

    return jsonify(health_status)


@app.route("/api/refresh-prompt", methods=["POST"])
def refresh_prompt():
    global avatar_llms
    data = request.get_json()
    avatar_id = data.get("avatar_id")
    if not avatar_id:
        return jsonify({"error": "avatar_id is required"}), 400

    print(f"Refresh prompt request received for avatar {avatar_id}.")

    avatars = json.load(open(avatars_path, "r"))
    avatar = next((a for a in avatars if a["id"] == avatar_id), None)
    if not avatar:
        return jsonify({"error": "Avatar not found."}), 404

    system_prompt_url = avatar.get("systemPromptUrl")
    if not system_prompt_url:
        return jsonify({"error": "Avatar does not have a systemPromptUrl."}), 400

    try:
        fetch_system_prompt_from_gdoc(avatar_id, system_prompt_url)
        # Reload the specific avatar's config
        llm, _, sensor_tool = generate_avatars_config(specific_avatar_id=avatar_id)
        if llm:
            avatar_llms[avatar_id] = llm
            avatar_sensor_tools[avatar_id] = sensor_tool
        else:
            return jsonify({"error": "Failed to reload avatar configuration."}), 500

        return jsonify({"status": "success", "message": f"Prompt for avatar {avatar_id} refreshed."})
    except Exception as e:
        print(f"Error refreshing prompt for avatar {avatar_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh-embeddings", methods=["POST"])
def refresh_embeddings():
    global avatar_rag_tools
    data = request.get_json()
    avatar_id = data.get("avatar_id")
    if not avatar_id:
        return jsonify({"error": "avatar_id is required"}), 400

    print(f"Refresh embeddings request received for avatar {avatar_id}.")

    avatars = json.load(open(avatars_path, "r"))
    avatar = next((a for a in avatars if a["id"] == avatar_id), None)
    if not avatar:
        return jsonify({"error": "Avatar not found."}), 404

    drive_folder_id = avatar.get("driveFolderId")
    if not drive_folder_id:
        return jsonify({"error": "Avatar does not have a driveFolderId."}), 400

    try:
        # Re-build the index and get the new query engines
        rag_tools = prepare_query_engines(avatar_id, drive_folder_id, refresh=True)
        avatar_rag_tools[avatar_id] = rag_tools
        return jsonify({"status": "success", "message": f"Embeddings for avatar {avatar_id} refreshed."})
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"Error refreshing embeddings for avatar {avatar_id}: {e}")
        return jsonify({"error": str(e)}), 500


# debate_general_prompt = "Right now you are on a deliberation-centered platform, debating with the user the topic of '{topic}'. In this mode you should always consider the best interests of the Lahn River. You must decide what the Lahn’s best interests are based on all of your context information. You are the Lahn’s advocate right now. Below is a brief description of the topic, which both you and the user have access to. You can present your position to the user as you answer questions they might have on the topic. '{description}'"
# topic_descriptions = {
#     'The Lahn should have legal personhood': "In recent years, rivers around the world have been granted legal personhood to recognize their intrinsic rights and protect their ecosystems. Granting the Lahn legal personhood would mean treating the river not merely as a resource but as a living entity with legal standing - analogous to the legal standing that a person or corporation holds. This shift could reshape how environmental protection is approached in the region, allowing for the river's interests to be formally represented in legal and political systems. And even create precedent for the river suing a company or the government, for example.",
#     'The Lahn should be able to own property': "If the Lahn were recognized as a legal person, it could theoretically hold property titles. This would allow the river to directly control land essential to its health—such as floodplains, wetlands, or riverbanks—ensuring its ecological integrity is not compromised by conflicting human interests. Property ownership could become a tool for the river to safeguard its own regeneration and future.",
#     'There should exist a “Lahn Fund”': "A dedicated “Lahn Fund” would serve as a financial mechanism to support the ongoing protection, restoration, and stewardship of the river. This fund could receive public and private contributions, fines from environmental damages, or a share of local economic activities that depend on the river. Managed in the river’s interest, the fund could finance ecological research, conservation projects, community engagement, and support the operational costs of the Avatar or legal guardianship system.",
#     'The Avatar should be able to legally speak on behalf of the Lahn': "The Lahn Avatar is envisioned as a voice for the river—an interface between natural and human systems. Allowing the Avatar to legally speak on behalf of the Lahn would formalize its role as a representative entity in decision-making processes. This could enable the river’s interests to be expressed in public hearings, governmental deliberations, and community forums, fostering a new model of ecological democracy and interspecies governance."
#   }


@app.route("/api/voice/web-search-toggle", methods=["POST"])
def web_search_toggle():
    """Toggle webSearchEnabled for an avatar (updates avatars.json)."""
    data = request.get_json()
    avatar_id = str(data.get("avatar_id", ""))
    enabled = bool(data.get("enabled", True))
    avatars = json.load(open(avatars_path, "r"))
    found = False
    for a in avatars:
        if a["id"] == avatar_id:
            if "llmDefaults" not in a:
                a["llmDefaults"] = {}
            a["llmDefaults"]["webSearchEnabled"] = enabled
            found = True
    if not found:
        return jsonify({"error": f"Avatar {avatar_id} not found"}), 404
    save_avatars(avatars, f"webSearchEnabled={enabled} for avatar {avatar_id}")
    return jsonify({"avatarId": avatar_id, "webSearchEnabled": enabled})


@app.route("/api/voice/web-search-status", methods=["GET"])
def web_search_status():
    avatar_id = request.args.get("avatar", "4")
    avatars = json.load(open(avatars_path, "r"))
    avatar = next((a for a in avatars if a["id"] == avatar_id), None)
    if not avatar:
        return jsonify({"error": "not found"}), 404
    admin_defaults = avatar.get("llmDefaults", {}) if avatar else {}
    return jsonify({"avatarId": avatar_id, "webSearchEnabled": admin_defaults.get("webSearchEnabled", True)})


@app.route("/api/chat", methods=["POST"])
def chat():
    # global llm, llm_choice, system_prompt
    _t0 = time.perf_counter()
    avatar_id = request.args.get("avatar", "0")
    data = request.get_json()

    # Load avatar to check for defaults
    avatars = json.load(open(avatars_path, "r"))
    avatar = next((a for a in avatars if a["id"] == avatar_id), None)
    avatar_name = avatar.get("name", "Unknown") if avatar else "Unknown"
    admin_defaults = avatar.get("llmDefaults", {}) if avatar else {}
    # Web search toggle: per-avatar Admin Default (llmDefaults.webSearchEnabled)
    web_search_enabled = bool(admin_defaults.get("webSearchEnabled", True))

    print(
        f"\n\n------------------------\nvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\nChat request received for Avatar '{avatar_name}' (id: {avatar_id})"
    )

    # Build user request params (with backward compat)
    user_params = {
        "chatProvider": data.get("chatProvider") or data.get("llmProvider"),
        "chatModel": data.get("chatModel") or data.get("llmModel"),
        "temperature": data.get("temperature"),
        "topK": data.get("topK"),
        "topP": data.get("topP"),
        "textQueryProvider": data.get("textQueryProvider"),
        "textQueryModel": data.get("textQueryModel"),
        "sensorProvider": data.get("sensorProvider"),
        "sensorModel": data.get("sensorModel"),
    }
    # Strip None values so resolve_llm_defaults treats them as unset
    user_params = {k: v for k, v in user_params.items() if v is not None}

    resolved, sources = resolve_llm_defaults(avatar_id, user_params)
    user_chat_provider = resolved["chat_provider"]
    user_chat_model = resolved["chat_model"]
    temperature = resolved["temperature"]
    top_k = resolved["top_k"]
    top_p = resolved["top_p"]
    text_query_provider = resolved["text_query_provider"]
    text_query_model = resolved["text_query_model"]
    sensor_provider = resolved["sensor_provider"]
    sensor_model = resolved["sensor_model"]

    # Reject requests for models known to be offline
    for role, model in [("chat", user_chat_model), ("text query", text_query_model), ("sensor", sensor_model)]:
        if _model_health_cache.get(model) == "offline":
            return jsonify({"error": f"The {role} model '{model}' is currently offline. Please select a different model in Avatar Garden."}), 503

    # Log resolved parameters with sources
    print("\n=== LLM Parameters (resolved) ===")
    print(f"Chat: provider={user_chat_provider} (from: {sources['chat_provider']}), model={user_chat_model} (from: {sources['chat_model']})")
    print(f"Temperature: {temperature} (from: {sources['temperature']})")
    print(f"Top K: {top_k} (from: {sources['top_k']})")
    print(f"Top P: {top_p} (from: {sources['top_p']})")
    print(f"Text Query: provider={text_query_provider} (from: {sources['text_query_provider']}), model={text_query_model} (from: {sources['text_query_model']})")
    print(f"Sensor: provider={sensor_provider} (from: {sources['sensor_provider']}), model={sensor_model} (from: {sources['sensor_model']})")
    print("=================================\n")

    # Get the avatar's system prompt from the pre-loaded config
    system_prompt = avatar_llms[avatar_id].system_prompt


    # if topic:
    #     print(f"→ Debate topic: {topic}")
    #     debate_prompt = debate_general_prompt.format(topic=topic, description=topic_descriptions[topic])
    #     # print('Prompt: ', debate_prompt)
    #     system_prompt_= system_prompt+ '\n' + debate_prompt
    # else:
    system_prompt_ = system_prompt

    chat_history = []

    conversation = data.get("history", "")
    print("Conversation history: ", conversation)

    chat_history = [
        {
            "role": "user" if m["sender"].lower() == "user" else "assistant",
            "content": m["text"],
        }
        for m in conversation
    ]

    if len(chat_history) == 0:
        return jsonify(
            {
                "error": 'Invalid request format. Ensure your payload has "history" in the format [{"sender": "User", "text": "content"}, ...]'
            }
        )

    chat_history.insert(0, {"role": "system", "content": system_prompt_})

    # === Create task-specific LLMs ===
    providers = json.load(open(LLM_PROVIDERS_PATH, "r"))

    # Text query LLM for keyword generation
    text_query_llm = create_llm_instance(
        text_query_provider, text_query_model, TEXT_QUERY_SYSTEM_PROMPT,
        task_name="text_query", providers=providers,
    )

    # === RAG Context Injection + Web Search Intent Classification ===
    # inject_rag_context returns (rag_injected, web_search_query, timings).
    # For RAG avatars the web search intent is detected in the same LLM call as keyword generation.
    # For no-RAG avatars a dedicated lightweight call is made instead.
    rag_injected, pre_web_search_query, timings = inject_rag_context(
        avatar_id, chat_history, conversation, text_query_llm, verbose=True,
        web_search_enabled=web_search_enabled,
    )
    pre_web_search_query = suppress_sensor_web_search(pre_web_search_query, avatar_id)
    # If this request triggered the avatar's RAG index cold load, surface it as
    # its own latency segment (otherwise it hides in "backend overhead").
    _load_ms = avatar_rag_tools.consume_load_ms(avatar_id)
    if _load_ms:
        timings["index_load_ms"] = _load_ms
    print(f"[TIMING] keyword-gen: {timings.get('keyword_gen_ms', 0)}ms, rag-retrieval: {timings.get('rag_retrieval_ms', 0)}ms")
    if rag_injected:
        print("Obtaining information for the LLM...")
        # Log the injected context (last user message now contains the RAG context)
        debug_log(avatar_id, "CHAT/RAG_CONTEXT", chat_history[-1]["content"])

    # === Sensor Tool Function Schema ===
    # NOTE: web_search is intentionally NOT listed as a callable function here.
    # Web search is handled proactively by the pre-classifier above (open-source models
    # do not tool-call reliably enough to be trusted with it at inference time).
    # Only analyze_sensor_data is kept as a model-invoked tool, because the always-fetch
    # snapshot already covers simple queries and complex trend analysis genuinely requires
    # the model to decide when to invoke the tool.
    sensor_config = avatar_sensor_tools.get(avatar_id)

    if sensor_config:
        print("Adding sensor tool function schema to prompt...")
        sensor_description = sensor_config.get('sensor_description', '')
        import json as json_module
        escaped_description = json_module.dumps(sensor_description, ensure_ascii=False)

        function_schema = f"""
FUNCTION:
You have access to this function. Call it ONLY when needed.

analyze_sensor_data(user_query="your question")
   - Call this for trend analysis or questions requiring historical sensor data.
   - Do NOT call this for simple current readings — those are already in the SENSOR SNAPSHOT above.

When you call the function, output ONLY the function call line, nothing else.
"""
        chat_history[0]["content"] += function_schema

    # === Always-fetch sensor snapshot (cached, 2-min TTL) ===
    # Gives LLM current readings for simple queries without tool invocation.
    # analyze_sensor_data() tool still handles complex analytical queries.
    if sensor_config:
        _ts = time.perf_counter()
        sensor_summary = _fetch_sensor_summary(avatar_id)
        timings["sensor_snapshot_ms"] = round((time.perf_counter() - _ts) * 1000)
        if sensor_summary:
            chat_history[0]["content"] += SENSOR_SNAPSHOT_INSTRUCTION.format(summary=sensor_summary)

    # === Web search tool schema (model-invoked, replaces pre-classifier) ===
    # The model decides when it genuinely needs external info (recent events,
    # news). The backend handles the tool call and re-invokes with labeled
    # results. Gated per-avatar via webSearchEnabled.
    if web_search_enabled:
        web_search_schema = """
FUNCTION:
You have access to this function. Call it ONLY when the user asks about
recent events, news, projects, or information that requires live internet
data (not covered by your knowledge base or sensor feed).

web_search(query="your search query")
   - Your query MUST include location terms: Morretes, Paraná, Rio Marumbi, Nhundiaquara.
   - Do NOT call for questions about river ecology, fauna, history, or community
     stories — those are already covered by the knowledge base context above.
   - Output ONLY the function call line, nothing else.
"""
        chat_history[0]["content"] += web_search_schema

    messages_to_send = chat_history

    # === Create chat LLM for main conversation ===
    llm = create_llm_instance(user_chat_provider, user_chat_model, system_prompt_,
                              temperature=temperature, top_k=top_k, top_p=top_p,
                              task_name="chat", providers=providers)

    _tllm = time.perf_counter()
    chat_completion = llm.complete(messages_to_send)
    response = chat_completion.text
    timings["main_llm_ms"] = round((time.perf_counter() - _tllm) * 1000)
    print(f"[TIMING] main-LLM: {time.perf_counter() - _tllm:.2f}s")

    print("\nAvatar response: ", response)
    debug_log(avatar_id, "CHAT/LLM_RESPONSE", response)
    debug_log(avatar_id, "CHAT/FULL_PROMPT", json.dumps(messages_to_send, ensure_ascii=False, indent=2))

    _tst = time.perf_counter()
    sensor_response, sensor_handled = handle_sensor_tool_call(
        response, avatar_id, sensor_provider, sensor_model, llm, chat_history, providers=providers)
    if sensor_handled:
        timings["sensor_tool_ms"] = round((time.perf_counter() - _tst) * 1000)
        timings["total_backend_ms"] = round((time.perf_counter() - _t0) * 1000)
        print(f"[TIMING] CHAT total: {time.perf_counter() - _t0:.2f}s")
        print("Avatar response after getting sensor data:", sensor_response)
        return jsonify({"reply": sensor_response, "timings": timings})

    if "analyze_sensor_data" in response and not avatar_sensor_tools.get(avatar_id):
        # Avatar has no sensor tool configured but model tried to call it
        print(f"No sensor tool available for avatar {avatar_id}")
        results = (
            "\nNote: Sensor data is not available for this avatar. "
            "Please ask about other aspects of the Lahn or choose a different avatar."
        )
        chat_completion_2 = llm.complete(
            chat_history + [{"role": "assistant", "content": results}],
        )
        timings["total_backend_ms"] = round((time.perf_counter() - _t0) * 1000)
        return jsonify({"reply": chat_completion_2.text.replace("*", ""), "timings": timings})

    # NOTE: There is intentionally no post-response web_search detection here.
    # Web search is handled entirely by the pre-classifier in inject_rag_context above.
    # Results are injected into context before the main LLM call, so the model never
    # needs to emit a tool call. This is the deliberate accommodation for open-source
    # models that do not tool-call as reliably as GPT-class models.

    response = response.replace("*", "")

    # === Web search tool call handling ===
    # Same pattern as analyze_sensor_data: if the model emitted web_search(query=...),
    # run Brave, re-invoke LLM with results clearly labeled as EXTERNAL source.
    # Bounded loop — a re-invoked answer may itself emit another tool call or
    # planning meta-text ("We should call web_search again..."), which leaked raw
    # to the user on 2026-09-12 (cutia conversation). Never display tool syntax.
    ws_rounds = 0
    while web_search_enabled and 'web_search(query="' in response and ws_rounds < 2:
        ws_rounds += 1
        _tw = time.perf_counter()
        q_start = response.find('web_search(query="') + len('web_search(query="')
        q_end = response.find('")', q_start)
        ws_query = response[q_start:q_end] if q_end > q_start else ""
        if not ws_query:
            break
        print(f"[web search] Tool-invoked query: {ws_query!r}")
        ws_results = web_search(ws_query, count=5, api_key=get_integration_key("BRAVE_SEARCH_API_KEY"))
        chat_history.append({"role": "assistant", "content": response})
        chat_history.append({"role": "user", "content":
            f"WEB SEARCH RESULTS (live internet data retrieved for you — may complement your "
            f"knowledge base):\n{ws_results}\n"
            "Use these results to inform your answer when they are relevant to the user's "
            "question about Morretes and the Rio Marumbi region. If they reference other "
            "locations, focus only on what applies here. Respond in the user's language, "
            "conversationally."
            + (" Do NOT output any function call or planning text — reply directly to the "
               "user now." if ws_rounds >= 2 else "")
        })
        re_completion = llm.complete(chat_history)
        response = re_completion.text.replace("*", "")
        timings["web_search_ms"] = round((time.perf_counter() - _tw) * 1000)
        print(f"[TIMING] web-search tool: {time.perf_counter() - _tw:.2f}s")
        debug_log(avatar_id, "VOICE/WEB_SEARCH_RESULTS", ws_results)
    if 'web_search(query="' in response:
        # Strip any surviving tool-call syntax so the user never sees it
        response = "\n".join(
            ln for ln in response.splitlines() if 'web_search(query="' not in ln
        ).strip()

    timings["total_backend_ms"] = round((time.perf_counter() - _t0) * 1000)
    print(f"[TIMING] CHAT total: {time.perf_counter() - _t0:.2f}s")
    return jsonify({"reply": response, "timings": timings})


@app.route("/api/debate-summary", methods=["POST"])
def debate_summary():
    print("Debate Summary request received.")
    data = request.get_json()
    conversation = data.get("history", "")
    topic = data.get("topic", "")
    summary = data.get("summary", "")

    formatted_history = format_history_as_string(conversation)

    prompt = f"""This is a debate between a human and an AI avatar for the Lahn river. Your job is to provide a summary outline in the format
            "Lahn:<Lahn's Central Perspective>\nPro:<Central Pro>\nCon:<Central Con of Lahn's perspective (deduced by you)>\n\nYou:<User's Central Perspective>\nPro:<Central Pro>\nCon:<Central Con of User's perspective (deduced by you)>", briefly outlining the Lahn's primary perspective, a pro and con of that perspective, the user's perspective
            and a pro and con of that as well. Keep all content very brief. You're summarizing, not re-iterating. You are provided with the most recent debate summary. If it already contains content, iterate on that content to reflect recent updates to the conversation.
            Topic being debated: {topic}

            Conversation:
            {formatted_history}

            Existing summary:
            {summary}

            Respond with an updated version of the summary in the described format. Make sure to preserve the specified formatting in the template "Lahn:\nPro:\nCon:\n\nYou:\nPro:\nCon:". No extra characters. The contents of your response should ba based purely on the given summary.
            Summaries for 'Lahn' and 'User'should be based purely on what they said. If any party is yet to contribute to the conversation, leave their summary blank, as in the template."""

    response = debate_summary_llm.complete(prompt)  # chat_engine.chat(prompt)
    # print('Summary model response: ', response)
    summary = str(response)  # .choices[0].message.content

    # chat_history = [
    #     ChatMessage(role="user" if m["sender"] == "user" else "assistant", content=m["text"])
    #     for m in conversation
    #     ]

    # print('User message:', prompt)
    # response = chat_engine.chat(messages=chat_history)
    print("Summary:", summary)

    return jsonify({"summary": summary})


# Would want to differentiate between general user and Admin uploads----
@app.route("/api/experience-upload", methods=["POST"])
def experience_upload():
    print("Experience upload received.")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

    os.makedirs(f"{UPLOAD_DIR}/text", exist_ok=True)

    # ---------- Save TEXT immediately ----------
    text = request.form.get("text", "")
    if text.strip():
        with open(f"{UPLOAD_DIR}/text/{timestamp}_message.txt", "w") as f:
            f.write(text.strip())

    # ---------- Save FILES (cheap, fast) ----------
    uploaded_files = request.files.getlist("files")
    saved_paths = []

    for f in uploaded_files:
        if not f or not f.filename:
            continue

        filename = secure_filename(f.filename)
        dest = os.path.join(UPLOAD_DIR, filename)
        f.save(dest)
        saved_paths.append(dest)

    # ---------- SAVE AUDIO but DO NOT TRANSCRIBE YET ----------
    audio_path = None

    if "audio" in request.files:
        audio_file = request.files["audio"]
        if audio_file and audio_file.filename:
            safe_name = secure_filename(audio_file.filename)
            ext = os.path.splitext(safe_name)[1]

            audio_path = os.path.join(UPLOAD_DIR, f"{timestamp}_audio{ext}")
            audio_file.save(audio_path)
            print("🎤 Audio saved, transcription deferred.")

    # ---------- FIRE BACKGROUND JOB ----------
    def background_job():
        try:
            if audio_path:
                transcript = transcribe_audio(audio_path)
                with open(f"{UPLOAD_DIR}/text/{timestamp}_transcript.txt", "w") as f:
                    f.write(transcript.strip())
                print("📝 Transcription saved.")

            refresh_embeddings()
            print("🔄 Embeddings refreshed.")
        except Exception as e:
            print("❌ Background job failed:", e)

    threading.Thread(target=background_job, daemon=True).start()

    # ---------- RETURN IMMEDIATELY ----------
    return jsonify(
        {"status": "success", "message": "Experience saved. Processing in background."}
    )


import subprocess


# Could pass audio_data directly to bypass file processing latency
@app.post("/api/voice-chat")
def voice_chat():
    print(
        "\n\n------------------------\nvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\nVoice Chat request received."
    )

    # Get system prompt from avatar config
    system_prompt = avatar_llms["0"].system_prompt

    # Do away with function-related instructions, for now.
    index = system_prompt.find(SENSOR_SECTION_MARKER)
    system_prompt_ = system_prompt[:index]

    try:
        # get uploaded file + pipeline choice
        audio_file = request.files["audio"]  # comes from FormData

        ext = audio_file.mimetype.split("/")[-1]
        audio_path = "data/temp." + ext
        audio_file.save(audio_path)

        pipeline = request.form.get("pipeline", "OpenAI gpt-realtime")

        print("Recieved voice chat request. Pipeline: ", pipeline)
        print("Audio: ", audio_path)

        output_wav = "data/temp." + ".wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, output_wav],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            print(f"[Success] Converted to: {output_wav}")
        except subprocess.CalledProcessError as e:
            print("[Error] ffmpeg decoding failed:\n", e.stderr.decode())
            return None

        # read into memory
        audio_bytes = output_wav  # audio_path #audio_file.read()

        # pass raw bytes to your dispatcher
        # Dispatch
        if pipeline == "OpenAI gpt-realtime":
            client = OpenAIRealtimeClient(
                OPENAI_API_KEY, model="gpt-realtime", prompt=system_prompt_,
                sensor_tool=avatar_sensor_tools.get("0")
            )
            response_audio, elapsed, cost_info = client.process_audio(audio_bytes)
        # elif pipeline == "OpenAI gpt4o":
        #     client = OpenAIRealtimeClient(OPENAI_API_KEY, model="gpt-4o-realtime-preview", prompt = system_prompt_)
        #     response_audio, elapsed, cost_info = client.process_audio(audio_bytes)
        # elif pipeline == "Cartesia":
        #     client = CartesiaOpenAIPipeline(CARTESIA_API_KEY, OPENAI_API_KEY, prompt = system_prompt_)
        #     response_audio, elapsed, cost_info = client.process_audio(audio_bytes)
        else:
            return jsonify({"error": f"Unknown pipeline '{pipeline}'"}), 400

        print("Done processing. Info: ", elapsed, cost_info)
        # print('System prompt: ', system_prompt_)

        # Convert raw PCM to WAV
        wav_bytes = pcm_to_wav_bytes(response_audio)

        # send audio directly
        return Response(wav_bytes, mimetype="audio/wav")

    except Exception as e:
        print({"error": str(e)})
        return jsonify({"error": str(e)}), 500


import json

from flask_sock import Sock

sock = Sock(app)


@sock.route("/api/voice-chat-stream")
def stream(ws):
    print(
        "\n\n------------------------\nvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\nStreaming Voice Chat request received."
    )

    system_prompt = avatar_llms["0"].system_prompt
    # Do away with function-related instructions, for now.
    index = system_prompt.find(SENSOR_SECTION_MARKER)
    system_prompt_ = system_prompt[:index]

    client = OpenAIRealtimeClient(
        OPENAI_API_KEY,
        model="gpt-realtime",
        prompt=system_prompt_,
        rag_tools=avatar_rag_tools["0"],
        sensor_tool=avatar_sensor_tools.get("0"),
        streaming=True,
        ws_client=ws,
    )
    client.connect_to_openai()

    #  gpt-4o-realtime-preview
    # Handle audio sent from frontend
    while True:
        msg = ws.receive()
        if msg is None:
            break

        try:
            data = json.loads(msg)
            if data.get("type") == "END":
                # conversation = data.get("conversation", [])
                # client.conversation_history = conversation
                client.commit_audio_buffer()
            else:
                # other control messages
                pass
        except json.JSONDecodeError:
            # raw base64 audio chunks
            client.append_audio(msg)


# Multi-Avatar Management


# avatars_api.py
from flask import Blueprint, jsonify, request

avatars_bp = Blueprint("avatars", __name__)


@avatars_bp.route("/api/avatars", methods=["GET", "POST"])
def avatars_collection():
    global avatar_llms, avatar_rag_tools
    """
    GET  /api/avatars  -> list all avatars
    POST /api/avatars  -> create a new avatar
    """

    avatars = json.load(open(avatars_path, "r"))
    print("\nAvatars: ", avatars)

    if request.method == "GET":
        # Frontend expects: [{ id, name, systemPromptUrl, contextDocsUrl, sensorApiUrl }, ...]
        # The pinned-default avatar (defaultAvatar: true) leads the list — both
        # the dropdown order and the avatar auto-selected on page load follow it.
        ordered = sorted(avatars, key=lambda a: not bool(a.get("defaultAvatar")))
        return jsonify(ordered), 200

    # POST
    data = request.get_json() or {}

    name = (data.get("name") or "").strip()
    system_prompt_url = data.get("systemPromptUrl") or ""
    context_docs_url = data.get("contextDocsUrl") or ""
    sensor_api_url = data.get("sensorApiUrl") or ""
    sensor_description = data.get("sensorDescription") or ""
    rag_languages = data.get("ragLanguages") or ""

    # Parse rag_languages: comma-separated string to list
    if rag_languages:
        rag_languages_list = [lang.strip() for lang in rag_languages.split(",") if lang.strip()]
    else:
        rag_languages_list = ["en", "de"]  # Default

    if not name:
        return jsonify({"error": "Avatar 'name' is required."}), 400

    print(
        "\n\n------------------------\nvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\nRecieved Request to Create New Avatar."
    )

    # Make id a string so it matches React's select value and find() comparison
    if len(avatars) == 0:
        next_id = "0"
    else:
        max_id = max(int(a["id"]) for a in avatars)
        next_id = str(max_id + 1)

    os.makedirs(f"avatars_context/{next_id}", exist_ok=True)

    if system_prompt_url != "":
        fetch_system_prompt_from_gdoc(next_id, system_prompt_url)

    # Run this in a thread
    # Set up RAG index for avatar
    if context_docs_url != "":
        drive_folder_id = context_docs_url.split("/folders/")[1].split("?")[0]
        print("Building Index for Avatar: " + next_id)
        # threading.Thread(target=build_or_load_index, kwargs={"avatar_id": next_id, "drive_folder_id": drive_folder_id}).start()
        results = build_or_load_index(next_id, drive_folder_id)
    else:
        drive_folder_id = None

    avatar = {
        "id": next_id,
        "name": name,
        "systemPromptUrl": system_prompt_url,
        "contextDocsUrl": context_docs_url,
        "driveFolderId": drive_folder_id,
        "sensorApiUrl": sensor_api_url,
        "sensorDescription": sensor_description,
        "ragLanguages": rag_languages_list,
    }

    avatars.append(avatar)
    print("Avatars after modification: ", avatars)
    save_avatars(avatars, reason=f"create avatar {next_id}")
    avatar_llms, avatar_rag_tools, avatar_sensor_tools = generate_avatars_config()
    # Frontend expects the created avatar object back
    return jsonify(avatar), 201


@avatars_bp.route("/api/avatars/<avatar_id>", methods=["PUT"])
def avatar_detail(avatar_id):
    """
    PUT /api/avatars/<avatar_id> -> update an existing avatar
    """
    data = request.get_json() or {}
    avatars = json.load(open(avatars_path, "r"))
    print("Avatars: ", avatars)

    # Find avatar by string id
    avatar = next((a for a in avatars if a["id"] == avatar_id), None)
    if avatar is None:
        return jsonify({"error": "Avatar not found."}), 404

    # Capture RAG-relevant state before applying updates (for change detection)
    old_rag_languages = list(avatar.get("ragLanguages") or [])
    old_pinned = bool(avatar.get("ragPinned"))

    # Only update fields present in request
    for field in ["name", "systemPromptUrl", "contextDocsUrl", "sensorApiUrl", "sensorDescription",
                  "ttsVoiceId", "ttsLanguage", "ragPinned", "keywordMode"]:
        if field in data and data[field] is not None:
            avatar[field] = data[field]
    # voiceBackend/voiceTransport live in llmDefaults (capability flags)
    if data.get("voiceBackend"):
        avatar.setdefault("llmDefaults", {})["voiceBackend"] = str(data["voiceBackend"]).lower()
    if data.get("voiceTransport") in ("agora", "rtp", "ws"):
        avatar.setdefault("llmDefaults", {})["voiceTransport"] = str(data["voiceTransport"]).lower()

    # Handle ragLanguages separately (comma-separated string to list)
    if "ragLanguages" in data and data["ragLanguages"] is not None:
        rag_languages = data["ragLanguages"]
        if rag_languages:
            avatar["ragLanguages"] = [lang.strip() for lang in rag_languages.split(",") if lang.strip()]
        else:
            avatar["ragLanguages"] = ["en", "de"]  # Default

    print("Avatars after modification: ", avatars)
    save_avatars(avatars, reason=f"update avatar {avatar_id}")

    # Surgical runtime update — touch only this avatar's state. A full
    # generate_avatars_config() here would replace the RAG cache and evict every
    # unpinned avatar's loaded index. (System prompts aren't refetched by PUT —
    # /api/refresh-prompt handles that — so avatar_llms needs no update.)
    avatar_sensor_tools[avatar_id] = sensor_config_from_avatar(avatar)
    if avatar.get("driveFolderId"):
        new_langs = avatar.get("ragLanguages", ["en", "de"])
        new_pinned = bool(avatar.get("ragPinned"))
        avatar_rag_tools.register(avatar_id, avatar["driveFolderId"], new_langs, pinned=new_pinned)
        if new_langs != old_rag_languages:
            # Retrieval language set changed — reload the index on next access
            avatar_rag_tools.invalidate(avatar_id)
        if new_pinned and not old_pinned:
            avatar_rag_tools.warm_pinned()
    return jsonify(avatar), 200


@avatars_bp.route("/api/avatars/<avatar_id>/set-default", methods=["POST"])
def avatar_set_default(avatar_id):
    """
    POST /api/avatars/<avatar_id>/set-default -> pin this avatar as the default:
    first in the dropdown and auto-selected on page load (survives refreshes).
    Unpins any previously pinned avatar. Returns the reordered avatar list.
    """
    avatars = json.load(open(avatars_path, "r"))
    avatar = next((a for a in avatars if a["id"] == avatar_id), None)
    if avatar is None:
        return jsonify({"error": "Avatar not found."}), 404

    for a in avatars:
        a["defaultAvatar"] = bool(a["id"] == avatar_id)
    save_avatars(avatars, reason=f"set default avatar to {avatar_id}")
    print(f"=== Default avatar set to '{avatar.get('name', avatar_id)}' (id: {avatar_id}) ===")
    return jsonify(sorted(avatars, key=lambda a: not bool(a.get("defaultAvatar")))), 200


@avatars_bp.route("/api/avatars/<avatar_id>/llm-defaults", methods=["POST", "DELETE"])
def avatar_llm_defaults(avatar_id):
    """
    POST   /api/avatars/<avatar_id>/llm-defaults -> save Admin defaults for avatar
    DELETE /api/avatars/<avatar_id>/llm-defaults -> clear Admin defaults for avatar
    """
    avatars = json.load(open(avatars_path, "r"))
    avatar = next((a for a in avatars if a["id"] == avatar_id), None)
    if avatar is None:
        return jsonify({"error": "Avatar not found."}), 404

    if request.method == "DELETE":
        # Clear defaults
        avatar_name = avatar.get("name", avatar_id)
        if "llmDefaults" in avatar:
            # webSearchEnabled is a capability flag, not a model default — it must
            # survive clearing Admin defaults. (2026-09-11: wiping the whole dict
            # re-enabled web search for avatars explicitly set to false.)
            preserved = {k: v for k, v in avatar["llmDefaults"].items()
                        if k == "webSearchEnabled"}
            if preserved:
                avatar["llmDefaults"] = preserved
            else:
                del avatar["llmDefaults"]
            print(f"\n=== Cleared Admin defaults for avatar '{avatar_name}' (id: {avatar_id}) ===\n")
        save_avatars(avatars, reason=f"clear llmDefaults for avatar {avatar_id}")
        return jsonify(avatar), 200

    # POST - save defaults
    data = request.get_json() or {}

    incoming = {
        "chat": {
            "provider": data.get("chatProvider"),
            "model": data.get("chatModel"),
            "temperature": data.get("temperature", 0.7),
            "top_k": data.get("topK", 40),
            "top_p": data.get("topP", 1.0)
        },
        "textQuery": {
            "provider": data.get("textQueryProvider"),
            "model": data.get("textQueryModel")
        },
        "sensor": {
            "provider": data.get("sensorProvider"),
            "model": data.get("sensorModel")
        },
        "voiceChat": {
            "provider": data.get("voiceChatProvider"),
            "model": data.get("voiceChatModel")
        }
    }

    # Voice backend routing is part of the Admin Defaults payload
    if data.get("voiceBackend"):
        incoming["voiceBackend"] = str(data["voiceBackend"]).lower()

    # Merge into existing defaults rather than replacing wholesale — a payload
    # missing a task's fields must not wipe that task (avatars.json lost
    # defaults twice on 2026-08-11 to partial/implicit saves; see ISSUES.md 13).
    llm_defaults = dict(avatar.get("llmDefaults", {}))
    for k in ("voiceBackend", "voiceTransport"):
        if k in incoming:
            llm_defaults[k] = incoming.pop(k)
    for task, cfg in incoming.items():
        cfg = {k: v for k, v in cfg.items() if v is not None}
        if cfg.get("model") or cfg.get("provider"):
            llm_defaults[task] = cfg

    avatar["llmDefaults"] = llm_defaults
    save_avatars(avatars, reason=f"save llmDefaults for avatar {avatar_id}")
    print(f"\n=== Saved Admin defaults for avatar '{avatar.get('name', avatar_id)}' (id: {avatar_id}) ===")
    print(f"{llm_defaults}")
    print("=================================\n")
    return jsonify(avatar), 200


app.register_blueprint(avatars_bp)


# LLM Providers Management
llm_providers_bp = Blueprint("llm_providers", __name__)

NON_CHAT_KEYWORDS = [
    "embed", "whisper", "tts", "rerank",       # common across providers
    "realtime", "audio", "transcribe",          # OpenAI audio/speech models
    "image", "sora",                            # OpenAI image/video generation
    "search",                                   # OpenAI search-specific models
    "codex",                                    # OpenAI code completion (non-chat API)
    "moderation",                               # OpenAI content moderation
    "computer-use",                             # OpenAI computer-use models
]

def _fetch_provider_models(provider):
    """Fetch available models from a provider's /v1/models endpoint."""
    api_base = provider.get("api_base", "").rstrip("/")
    api_key_env = provider.get("api_key_env")
    api_key = provider.get("api_key", "") or (os.getenv(api_key_env) if api_key_env else None)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Some providers (e.g. OpenAI) already include /v1 in their api_base
    if api_base.endswith("/v1"):
        url = f"{api_base}/models"
    else:
        url = f"{api_base}/v1/models"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    models = [m["id"] for m in data.get("data", [])]
    # Filter out non-chat models (embeddings, speech, rerankers)
    return [m for m in models if not any(kw in m.lower() for kw in NON_CHAT_KEYWORDS)]


def _mask_provider(p):
    p = dict(p)
    # Check JSON-stored key first, then env var fallback (mirrors create_llm_instance)
    key = p.pop("api_key", "") or (os.getenv(p.get("api_key_env", ""), "") if p.get("api_key_env") else "")
    p["key_set"] = bool(key)
    p["key_last4"] = key[-4:] if key else ""
    return p


@llm_providers_bp.route("/api/llm-providers", methods=["GET", "POST"])
def llm_providers_collection():
    """
    GET /api/llm-providers -> list all LLM providers
    POST /api/llm-providers -> create a new LLM provider
    """
    providers = json.load(open(LLM_PROVIDERS_PATH, "r"))

    if request.method == "GET":
        return jsonify([_mask_provider(p) for p in providers]), 200

    # POST
    data = request.get_json() or {}

    provider_id = (data.get("id") or "").strip()
    name = (data.get("name") or "").strip()
    provider_key = (data.get("provider_key") or "custom").strip()
    api_base = data.get("api_base") or ""
    api_key = data.get("api_key") or ""
    models = data.get("models") or []

    if not name:
        return jsonify({"error": "Provider 'name' is required."}), 400

    if not provider_id:
        # Generate ID from name if not provided
        provider_id = name.lower().replace(" ", "-").replace("/", "-")

    # Check if ID already exists
    if any(p["id"] == provider_id for p in providers):
        return jsonify({"error": f"Provider with id '{provider_id}' already exists."}), 400

    if not models:
        return jsonify({"error": "At least one model is required."}), 400

    print(f"\n\n------------------------\nvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\nReceived Request to Create New LLM Provider: {name}")

    provider = {
        "id": provider_id,
        "name": name,
        "provider_key": provider_key,
        "api_base": api_base,
        "models": models if isinstance(models, list) else [m.strip() for m in models.split(",")],
    }

    if api_key:
        provider["api_key"] = api_key

    providers.append(provider)
    print("LLM Providers after modification: ", providers)
    json.dump(providers, open(LLM_PROVIDERS_PATH, "w"))
    return jsonify(_mask_provider(provider)), 201


@llm_providers_bp.route("/api/llm-providers/<provider_id>", methods=["PUT", "DELETE"])
def llm_provider_detail(provider_id):
    """
    PUT /api/llm-providers/<provider_id> -> update an existing LLM provider
    DELETE /api/llm-providers/<provider_id> -> delete an LLM provider
    """
    providers = json.load(open(LLM_PROVIDERS_PATH, "r"))
    provider = next((p for p in providers if p["id"] == provider_id), None)

    if not provider:
        return jsonify({"error": "Provider not found."}), 404

    if request.method == "DELETE":
        providers = [p for p in providers if p["id"] != provider_id]
        json.dump(providers, open(LLM_PROVIDERS_PATH, "w"))
        return jsonify({"status": "success", "message": f"Provider {provider_id} deleted."}), 200

    # PUT
    data = request.get_json() or {}

    # Handle API key update — store directly in provider JSON, never write to .env
    if "api_key" in data and data["api_key"] is not None:
        api_key = data["api_key"]
        if api_key:
            provider["api_key"] = api_key
            print(f"API key stored in llm_providers.json for {provider_id}")

    # Update other fields (except api_key which we handle above)
    for field in ["name", "provider_key", "api_base", "models"]:
        if field in data and data[field] is not None:
            provider[field] = data[field]

    print("LLM Providers after modification: ", providers)
    json.dump(providers, open(LLM_PROVIDERS_PATH, "w"))
    return jsonify(_mask_provider(provider)), 200


@llm_providers_bp.route("/api/llm-providers/<provider_id>/refresh-models", methods=["POST"])
def refresh_provider_models(provider_id):
    """
    POST /api/llm-providers/<provider_id>/refresh-models
    Fetches the live model list from the provider's /v1/models endpoint
    and persists it to llm_providers.json.
    """
    providers = json.load(open(LLM_PROVIDERS_PATH, "r"))
    provider = next((p for p in providers if p["id"] == provider_id), None)

    if not provider:
        return jsonify({"error": "Provider not found."}), 404

    try:
        models = _fetch_provider_models(provider)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch models from provider: {str(e)}"}), 502

    global _models_last_refreshed
    provider["models"] = models
    json.dump(providers, open(LLM_PROVIDERS_PATH, "w"), indent=2)
    _models_last_refreshed = datetime.now().isoformat(timespec="seconds")
    print(f"Refreshed models for provider '{provider_id}': {models}")
    return jsonify({"models": models}), 200


@llm_providers_bp.route("/api/llm-options", methods=["GET"])
def llm_options():
    """
    GET /api/llm-options -> returns providers in dropdown-friendly format
    Format: { "options": { "provider_id": { "name": "...", "models": [...] } }, "last_refreshed": "..." }
    """
    providers = json.load(open(LLM_PROVIDERS_PATH, "r"))
    options = {}
    for p in providers:
        if p.get("hidden"):
            continue
        options[p["id"]] = {
            "name": p["name"],
            "models": p["models"],
        }
    return jsonify({"options": options, "last_refreshed": _models_last_refreshed}), 200


@llm_providers_bp.route("/api/integrations", methods=["GET"])
def get_integrations():
    data = _load_integrations()
    result = {}
    for k in INTEGRATION_KEYS:
        val = data.get(k, "") or os.getenv(k, "")
        result[k] = {
            "set": bool(val),
            "last4": val[-4:] if val else "",
            "source": "file" if data.get(k) else ("env" if os.getenv(k) else "unset"),
        }
    return jsonify(result)


@llm_providers_bp.route("/api/integrations", methods=["PUT"])
def update_integrations():
    updates = request.get_json() or {}
    data = _load_integrations()
    changed = []
    for k in INTEGRATION_KEYS:
        if k in updates and updates[k]:  # only save non-empty values
            data[k] = updates[k]
            changed.append(k)
    if changed:
        _save_integrations(data)
        print(f"[integrations] Updated keys: {changed}")
    return jsonify({"updated": changed})


app.register_blueprint(llm_providers_bp)


import re
import socket

ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
FLASK_WARN = 'WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.'


@app.route("/api/backend-log", methods=["GET"])
def backend_log():
    log_path = os.path.expanduser("~/backend_log.0")
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
        # Keep last 4000 lines
        lines = lines[-4000:]
        # Strip ANSI escape codes for browser display
        lines = [ANSI_RE.sub("", line) for line in lines]
        # Filter out the Flask dev server warning
        lines = [line for line in lines if FLASK_WARN not in line]
        return jsonify({"log": "".join(lines)})
    except FileNotFoundError:
        return jsonify({"error": "Log file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _auto_refresh_models():
    """Background thread: refresh all provider model lists every 24 hours."""
    import time
    global _models_last_refreshed
    REFRESH_INTERVAL = 24 * 60 * 60  # seconds

    while True:
        try:
            providers = json.load(open(LLM_PROVIDERS_PATH, "r"))
            for provider in providers:
                if provider.get("hidden"):
                    continue
                try:
                    models = _fetch_provider_models(provider)
                    provider["models"] = models
                    print(f"[model refresh] {provider['id']}: {len(models)} models")
                except Exception as e:
                    print(f"[model refresh] {provider['id']}: failed — {e}")
            json.dump(providers, open(LLM_PROVIDERS_PATH, "w"), indent=2)
            _models_last_refreshed = datetime.now().isoformat(timespec="seconds")
        except Exception as e:
            print(f"[model refresh] Could not read providers file: {e}")

        time.sleep(REFRESH_INTERVAL)


threading.Thread(target=_auto_refresh_models, daemon=True).start()


# ═══════════════════════════════════════════════════════════
# Voice Avatar Garden — Agora Conversational AI integration
# ═══════════════════════════════════════════════════════════
import uuid as _uuid
import base64 as _base64
from urllib.parse import quote as _urlquote

voice_bp = Blueprint("voice", __name__)

AGORA_CONV_AI_BASE = "https://api.agora.io/api/conversational-ai-agent/v2/projects"

# Default TTS voice/language — Cartesia "Miles - Yogi", an English voice.
# Per-avatar override via avatars.json `ttsVoiceId` / `ttsLanguage`: voices are
# accent-native, so avatars whose primary language isn't English should set a
# native voice (verified 2026-08-11: an EN voice anglicizes PT proper nouns —
# "Morretes"→"Moretz" — while a native PT voice renders them perfectly).
DEFAULT_TTS_VOICE_ID = "f114a467-c40a-4db8-964d-aaba89cd08fa"
DEFAULT_TTS_LANGUAGE = "en"

# Most recent voice request timings, keyed by avatar_id. The Agora agent calls
# /api/voice/chat-completions directly, so the browser can't read timings off that
# response — it fetches them from /api/voice/last-timings instead.
_last_voice_timings = {}

# Sentence boundary for streaming flushes: terminal punctuation (optionally
# followed by closing quotes/brackets) plus trailing whitespace, or a newline.
_SENTENCE_END_RE = re.compile(r'[.!?…]["\'”’)\]]*\s+|\n+')


def _agora_auth_headers():
    creds = _base64.b64encode(f"{get_integration_key('AGORA_CUSTOMER_ID')}:{get_integration_key('AGORA_CUSTOMER_SECRET')}".encode()).decode()
    return {"Authorization": f"Basic {creds}", "Content-Type": "application/json"}


@voice_bp.route("/api/voice/token", methods=["GET"])
def get_voice_token():
    """Return Agora App ID and RTC token for the client to join a channel."""
    channel = request.args.get("channel", "avatar-lab")
    uid = int(request.args.get("uid", 0))
    # Token generation requires agora-token-builder + AGORA_APP_CERTIFICATE.
    # In dev (no certificate), pass token=None — Agora allows this in test mode.
    # TODO: install agora-token-builder and generate a real token when certificate is set.
    app_certificate = get_integration_key("AGORA_APP_CERTIFICATE")
    agora_app_id = get_integration_key("AGORA_APP_ID")
    token = None
    if app_certificate:
        try:
            import time as _time
            from agora_token_builder import RtcTokenBuilder
            from agora_token_builder.RtcTokenBuilder import Role_Publisher
            token = RtcTokenBuilder.buildTokenWithUid(
                agora_app_id, app_certificate,
                channel, uid, Role_Publisher,
                int(_time.time()) + 3600
            )
        except ImportError:
            print("[Voice] agora-token-builder not installed — using null token")

    return jsonify({"appId": agora_app_id, "channel": channel, "uid": uid, "token": token})


# --- ConvoAI reaper: lingering agents burn $0.10/min (2026-09-15 bill:
# 148 paid minutes). Every agent/start registers {agentId: channel, started};
# the voice callback timestamps the channel's last turn; a thread stops any
# agent idle > REAPER_MINUTES. Survives backend restarts via convo_agents.json.
try:
    _convo_agents = json.load(open(_convo_registry_path, "r"))
except Exception:
    _convo_agents = {}
CONVO_REAPER_MINUTES = 10
CONVO_REAPER_TICK_S = 60
_convo_agents = {}        # agent_id -> {"channel", "started"}
_convo_last_turn = {}     # channel -> monotonic timestamp of last LLM turn
_convo_registry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "convo_agents.json")


def _convo_register(agent_id, channel):
    _convo_agents[agent_id] = {"channel": channel, "started": time.time()}
    try:
        json.dump(_convo_agents, open(_convo_registry_path, "w"))
    except Exception as e:
        print(f"[reaper] registry save failed: {e}")


def _convo_unregister(agent_id):
    _convo_agents.pop(agent_id, None)
    try:
        json.dump(_convo_agents, open(_convo_registry_path, "w"))
    except Exception:
        pass


def _convo_reaper_loop():
    while True:
        time.sleep(CONVO_REAPER_TICK_S)
        # merge external registrations (device/backend spawns in other processes,
        # manual registry edits) — the file is the source of truth
        try:
            reg = json.load(open(_convo_registry_path, "r"))
            for k, v in reg.items():
                _convo_agents.setdefault(k, v)
        except Exception:
            pass
        now = time.time()
        if _convo_agents:
            print(f"[reaper] tick: {len(_convo_agents)} agent(s) tracked")
        for agent_id, info in list(_convo_agents.items()):
            last_turn = _convo_last_turn.get(info["channel"], 0)
            last = max(info["started"], last_turn)
            if now - last > CONVO_REAPER_MINUTES * 60:
                print(f"[reaper] stopping idle ConvoAI agent {agent_id} "
                      f"(channel {info['channel']}, idle > {CONVO_REAPER_MINUTES} min)")
                try:
                    requests.post(
                        f"{AGORA_CONV_AI_BASE}/{get_integration_key('AGORA_APP_ID')}/agents/{agent_id}/leave",
                        headers=_agora_auth_headers(), timeout=15)
                except Exception as e:
                    print(f"[reaper] leave failed: {e}")
                _convo_unregister(agent_id)


print(f"[reaper] active: tick {CONVO_REAPER_TICK_S}s, idle threshold {CONVO_REAPER_MINUTES}min, "
          f"{len(_convo_agents)} agents tracked")
threading.Thread(target=_convo_reaper_loop, daemon=True).start()


@voice_bp.route("/api/voice/agent/start", methods=["POST"])
def start_voice_agent():
    """Start a voice agent for the given avatar and channel.

    Routed by the avatar's voiceBackend (admin default, LLM Config -> Voice):
    - "convoai" (default): Agora Conversational AI agent (cloud, uid 999)
    - "vps": self-hosted orchestrator session (agora-python-server-sdk, uid 777,
      same LLM brain via localhost callback) — Fase 5 routing
    """
    data = request.get_json() or {}
    avatar_id = str(data.get("avatarId", "0"))
    channel = data.get("channel", "avatar-lab")
    user_uid = data.get("userUid", 0)
    streaming = bool(data.get("streaming"))
    # Optional per-request agent parameters (e.g. output_audio_codec for the
    # ReSpeaker hardware client). Web clients omit it -> payload unchanged.
    agent_parameters = data.get("parameters")

    if avatar_id not in avatar_llms:
        return jsonify({"error": f"Unknown avatar: {avatar_id}"}), 404

    # Load admin defaults to determine TTS voice etc.
    avatars = json.load(open(avatars_path, "r"))
    avatar = next((a for a in avatars if a["id"] == avatar_id), None)

    voice_backend = str((avatar or {}).get("llmDefaults", {}).get(
        "voiceBackend", "convoai")).lower()
    llm_d = (avatar or {}).get("llmDefaults", {})
    if voice_backend == "vps":
        # Web clients omit `parameters` (no output_audio_codec) — they have
        # browser AEC, so full-duplex + barge-in is safe, and they always ride
        # the Agora transport (browsers speak WebRTC, not raw RTP). Device
        # clients send parameters — they honor the avatar's transport choice.
        is_device = isinstance(agent_parameters, dict) and bool(agent_parameters)
        full_duplex = not is_device
        voice_transport = str(llm_d.get("voiceTransport", "agora")).lower()
        if voice_transport == "ws":
            return _start_orchestrator_session(avatar_id, channel, True,
                                              codec="pcm", transport="ws")
        transport = "rtp" if (is_device and voice_transport == "rtp") else "agora"
        return _start_orchestrator_session(avatar_id, channel, full_duplex,
                                          codec=("opus" if full_duplex else "g722"),
                                          transport=transport,
                                          device_uid=user_uid)

    # Use avatar system prompt — trim sensor/function instructions (handled separately)
    system_prompt = avatar_llms[avatar_id].system_prompt
    cutoff = system_prompt.find(SENSOR_SECTION_MARKER)
    if cutoff > 0:
        system_prompt = system_prompt[:cutoff].strip()

    # Generate RTC token for the agent's own UID so it can join the channel
    agent_uid = 999
    agent_token = ""
    agora_app_id = get_integration_key("AGORA_APP_ID")
    app_certificate = get_integration_key("AGORA_APP_CERTIFICATE")
    if app_certificate:
        try:
            import time as _time
            from agora_token_builder import RtcTokenBuilder
            from agora_token_builder.RtcTokenBuilder import Role_Publisher
            agent_token = RtcTokenBuilder.buildTokenWithUid(
                agora_app_id, app_certificate,
                channel, agent_uid, Role_Publisher,
                int(_time.time()) + 3600
            )
        except ImportError:
            print("[Voice] agora-token-builder not installed — agent token will be empty")

    agent_payload = {
        "name": f"avatargarden-avatar-{avatar_id}-{channel}",
        "properties": {
            **({"parameters": agent_parameters} if isinstance(agent_parameters, dict) and agent_parameters else {}),
            "channel": channel,
            "token": agent_token,
            "agent_rtc_uid": str(agent_uid),
            "remote_rtc_uids": [str(user_uid)] if user_uid else ["*"],
            "idle_timeout": 120,
            "asr": {
                "vendor": "deepgram",
                "params": {
                    "api_key": get_integration_key("DEEPGRAM_API_KEY"),
                    "model": "nova-3",
                    # Multilingual code-switching. Verified 2026-08-11 (isolated
                    # Deepgram test): fixes PT/DE transcription, English unaffected.
                    # Not in Agora's documented language list, but Agora passes
                    # params through to Deepgram unvalidated.
                    "language": "multi",
                    # nova-3 keyterm prompting — boost the avatar's proper nouns
                    # (residual failure class: "Rio Sagrado"→"risigrado").
                    **({"keyterm": _urlquote(avatar["name"])} if avatar and avatar.get("name") else {}),
                }
            },
            "llm": {
                "url": f"{BACKEND_PUBLIC_URL}/api/voice/chat-completions?avatar={avatar_id}"
                       + f"&channel={channel}"
                       + ("&streaming=1" if streaming else ""),
                "vendor": "custom",
                "style": "openai",
                "params": {"model": "avatar"},
                "system_messages": [{"role": "system", "content": system_prompt}],
                "failure_message": "I need a moment. Please hold on.",
                "max_history": 10,
                "timeout_ms": 30000,
            },
            "tts": {
                "vendor": "cartesia",
                "params": {
                    "api_key": get_integration_key("CARTESIA_API_KEY"),
                    # Explicit model choice (2026-08-11): sonic-3 — current
                    # Cartesia generation, better multilingual/prosody than the
                    # sonic-2 the integration example shipped with. Verified
                    # working on our API key.
                    "model_id": "sonic-3",
                    # Documented Cartesia voice shape — a flat "voice_id" key is
                    # silently ignored and Cartesia falls back to a default voice.
                    "voice": {
                        "mode": "id",
                        "id": (avatar.get("ttsVoiceId") if avatar else None) or DEFAULT_TTS_VOICE_ID,
                    },
                    # 12/Sep: explicit 16 kHz mono PCM output — Cartesia's
                    # default (44.1 kHz) played through the G722/16 kHz RTC
                    # pipeline sounds ~2.8x fast (chipmunk). Mirrors the
                    # validated start_g722_agent.py payload.
                    "output_format": {
                        "container": "raw",
                        "encoding": "pcm_s16le",
                        "sample_rate": 16000,
                    },
                    # Synthesis language hint (Cartesia defaults to "en" when
                    # absent, even for non-English reply text).
                    "language": (avatar.get("ttsLanguage") if avatar else None) or DEFAULT_TTS_LANGUAGE,
                }
            }
        }
    }

    print(f"[Voice] Starting agent for avatar {avatar_id} on channel {channel}")
    resp = requests.post(
        f"{AGORA_CONV_AI_BASE}/{agora_app_id}/join",
        headers=_agora_auth_headers(),
        json=agent_payload,
        timeout=15,
    )

    if resp.status_code >= 400:
        print(f"[Voice] Agora agent start failed: {resp.status_code} — {resp.text}")
        try:
            return jsonify({"error": resp.json()}), resp.status_code
        except Exception:
            return jsonify({"error": resp.text}), resp.status_code

    agent_data = resp.json()
    agent_id = agent_data.get("agent_id") or agent_data.get("id")
    print(f"[Voice] Agent started: {agent_id}")
    _convo_register(agent_id, channel)
    return jsonify({"agentId": agent_id, "channel": channel})


# --- VPS-hosted orchestrator sessions (voiceBackend: "vps", Fase 5) ---
ORCH_DIR = "/root/AvatarGarden/orchestrator"
ORCH_PY = "/root/agora-orchestrator-venv/bin/python"
ORCH_LOG_DIR = "/root/AvatarGarden/orchestrator/logs"
_orch_sessions = {}  # channel -> {"pid", "agent_id", "avatar_id", "started", ...}
ORCH_RTP_PORT_BASE = 26100
ORCH_RTP_PORT_MAX = 26999

def _session_alive(pid):
    """Registry entries can be stale (crashed sessions, pid reuse) — verify
    the process identity via /proc cmdline, not just os.kill."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"orchestrator_main" in f.read()
    except OSError:
        return False


def _alloc_rtp_port():
    used = set()
    for s in list(_orch_sessions.values()):
        if "rtp_port" not in s:
            continue
        if _session_alive(s["pid"]):
            used.add(s["rtp_port"])
    p = ORCH_RTP_PORT_BASE
    while p <= ORCH_RTP_PORT_MAX:
        if p not in used:
            return p
        p += 1
    raise RuntimeError("no free RTP ports")


def _start_orchestrator_session(avatar_id, channel, full_duplex=False, codec="g722",
                               transport="agora", device_uid=1):
    """Spawn a VoiceSession process for this channel; adopt an existing one."""
    existing = _orch_sessions.get(channel)
    if existing:
        try:
            os.kill(existing["pid"], 0)
            print(f"[Voice] Orchestrator already live for {channel} "
                  f"(pid {existing['pid']}) — adopt")
            resp = {"agentId": existing["agent_id"], "channel": channel,
                    "transport": existing.get("transport", "agora")}
            if resp["transport"] == "ws":
                resp["wsPath"] = "/voice-ws"
            if "rtp_port" in existing:
                from urllib.parse import urlparse
                host = urlparse(os.environ.get("BACKEND_PUBLIC_URL", "")).hostname
                resp.update({"rtpHost": socket.gethostbyname(host) if host else "127.0.0.1",
                             "rtpPort": existing["rtp_port"],
                             "deviceUid": existing.get("device_uid", 1)})
            return jsonify(resp)
        except OSError:
            _orch_sessions.pop(channel, None)
    os.makedirs(ORCH_LOG_DIR, exist_ok=True)
    log_f = open(os.path.join(ORCH_LOG_DIR, f"{channel}.log"), "ab")
    cmd = [ORCH_PY, "-u", "orchestrator_main.py",
           "--avatar", str(avatar_id), "--channel", channel,
           "--idle", "180", "--max", "1800"]
    if full_duplex:
        cmd.append("--full-duplex")
    if codec in ("g722", "opus"):
        cmd += ["--codec", codec]  # ws mode takes no codec (raw PCM)
    rtp_port = None
    if transport == "rtp":
        rtp_port = _alloc_rtp_port()
        cmd += ["--rtp-port", str(rtp_port)]
    if transport == "ws":
        for s in list(_orch_sessions.values()):
            if s.get("transport") == "ws" and _session_alive(s["pid"]):
                return jsonify({"error": "another WebSocket voice session is live",
                                "agentId": s["agent_id"], "channel": "ws"}), 409
        cmd += ["--ws-port", "8010"]
    proc = subprocess.Popen(
        cmd, cwd=ORCH_DIR, stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True)
    agent_id = f"vps-{proc.pid}"
    entry = {"pid": proc.pid, "agent_id": agent_id,
             "avatar_id": avatar_id, "started": time.time(),
             "transport": transport}
    if rtp_port:
        entry["rtp_port"] = rtp_port
    _orch_sessions[channel] = entry
    print(f"[Voice] VPS orchestrator started: channel={channel} pid={proc.pid} "
          f"transport={transport}{' rtp_port=' + str(rtp_port) if rtp_port else ''}")
    resp = {"agentId": agent_id, "channel": channel, "transport": transport}
    if rtp_port:
        from urllib.parse import urlparse
        host = urlparse(os.environ.get("BACKEND_PUBLIC_URL", "")).hostname
        resp["rtpHost"] = socket.gethostbyname(host) if host else "127.0.0.1"
        resp["rtpPort"] = rtp_port
        resp["deviceUid"] = device_uid
    if transport == "ws":
        resp["wsPath"] = "/voice-ws"
    return jsonify(resp)


@voice_bp.route("/api/voice/agent/stop", methods=["POST"])
def stop_voice_agent():
    """Stop a running voice agent (Agora ConvoAI or a VPS orchestrator session)."""
    data = request.get_json() or {}
    agent_id = data.get("agentId")
    if not agent_id:
        return jsonify({"error": "agentId required"}), 400

    # VPS orchestrator sessions: agentId "vps-<pid>" (registry keeps the map)
    if str(agent_id).startswith("vps-"):
        pid = None
        try:
            pid = int(str(agent_id)[4:])
        except ValueError:
            for ch, s in list(_orch_sessions.items()):
                if s["agent_id"] == agent_id:
                    pid = s["pid"]
                    _orch_sessions.pop(ch, None)
                    break
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"[Voice] VPS orchestrator stopped: pid {pid}")
            except OSError as e:
                print(f"[Voice] VPS orchestrator stop failed: {e}")
        for ch, s in list(_orch_sessions.items()):
            if s["pid"] == pid:
                _orch_sessions.pop(ch, None)
        return jsonify({"status": "stopped"})

    resp = requests.post(
        f"{AGORA_CONV_AI_BASE}/{get_integration_key('AGORA_APP_ID')}/agents/{agent_id}/leave",
        headers=_agora_auth_headers(),
        timeout=15,
    )
    print(f"[Voice] Agent stopped: {agent_id} (status {resp.status_code})")
    return jsonify({"status": "stopped"})


@voice_bp.route("/api/voice/tts-voices", methods=["GET"])
def list_tts_voices():
    """List available Cartesia voices (id, name, language) for the AvatarGarden voice picker."""
    try:
        resp = requests.get(
            "https://api.cartesia.ai/voices/?limit=100",
            headers={"X-API-Key": get_integration_key("CARTESIA_API_KEY"),
                     "Cartesia-Version": "2025-04-16"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        voices = data if isinstance(data, list) else data.get("data", [])
        trimmed = sorted(
            [{"id": v["id"], "name": v.get("name", ""), "language": v.get("language", "")}
             for v in voices],
            key=lambda v: (v["language"], v["name"]),
        )
        return jsonify(trimmed)
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@voice_bp.route("/api/voice/last-timings", methods=["GET"])
def voice_last_timings():
    """Return backend timings for the most recent voice request of an avatar."""
    avatar_id = str(request.args.get("avatar", "0"))
    return jsonify(_last_voice_timings.get(avatar_id) or {})


@voice_bp.route("/api/voice/chat-completions", methods=["POST"])
def voice_chat_completions():
    """
    OpenAI-compatible SSE endpoint called by the Agora Conversational AI agent.
    Runs the full RAG + sensor + LLM pipeline and streams back the response.
    """
    from flask import stream_with_context

    avatar_id = str(request.args.get("avatar", "0"))
    _chan = request.args.get("channel")
    if _chan:
        _convo_last_turn[_chan] = time.time()
    data = request.get_json() or {}
    messages = data.get("messages", [])

    if avatar_id not in avatar_llms:
        return jsonify({"error": f"Unknown avatar: {avatar_id}"}), 404

    # Agora includes the system prompt we configured — skip it, use ours from avatar config
    conversation_msgs = [m for m in messages if m["role"] != "system"]
    if not conversation_msgs:
        return jsonify({"error": "No conversation messages"}), 400

    print(f"\n[Voice] Chat completion — avatar={avatar_id}, turns={len(conversation_msgs)}")

    # Resolve LLM parameters from admin defaults (no user params for voice)
    resolved, _ = resolve_llm_defaults(avatar_id)
    chat_provider, chat_model = fallback_model(resolved["voice_chat_provider"], resolved["voice_chat_model"])
    text_query_provider_raw, text_query_model_raw = fallback_model(resolved["text_query_provider"], resolved["text_query_model"])
    temperature = resolved["temperature"]
    top_k = resolved["top_k"]
    top_p = resolved["top_p"]
    text_query_provider = text_query_provider_raw
    text_query_model = text_query_model_raw
    sensor_provider, sensor_model = fallback_model(resolved["sensor_provider"], resolved["sensor_model"])

    system_prompt = avatar_llms[avatar_id].system_prompt
    providers = json.load(open(LLM_PROVIDERS_PATH, "r"))

    # Build chat history with system prompt
    chat_history = [{"role": "system", "content": system_prompt}]
    chat_history += [
        {"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
        for m in conversation_msgs
    ]

    # Conversation in the format RAG helpers expect
    conversation_for_rag = [
        {"sender": "user" if m["role"] == "user" else "avatar", "text": m["content"]}
        for m in conversation_msgs
    ]

    _vt0 = time.perf_counter()
    vtimings = {}

    # RAG context injection
    # Voice RAG: LLM-based keyword generation (same path as chat) — context-aware
    # across turns and robust on short/ambiguous utterances. The configured
    # text-query model should be small (e.g. 8B, ~1s) to fit Agora's timeout.
    # The web-search intent field from this call is unused — voice web search
    # is model-invoked (VOICE_WEB_SEARCH_SCHEMA) and gated by webSearchEnabled.
    try:
        if avatar_rag_tools.get(avatar_id):
            rag_languages = avatar_rag_tools[avatar_id][4] if len(avatar_rag_tools[avatar_id]) > 4 else ['en', 'de']
            text_query_llm = create_llm_instance(
                text_query_provider, text_query_model, TEXT_QUERY_SYSTEM_PROMPT,
                task_name="voice_text_query", providers=providers,
            )
            _tr = time.perf_counter()
            if _avatar_keyword_mode(avatar_id) == "local":
                try:
                    last_user = next(
                        (m.get("text", "") for m in reversed(conversation_for_rag)
                         if str(m.get("sender", "")).lower() == "user"),
                        "",
                    )
                    keywords_by_lang = extract_keywords_multilingual(last_user, rag_languages) if last_user.strip() else {}
                    _web_q = None
                    if not any(keywords_by_lang.values()):
                        raise ValueError("VPS keyword extraction produced no keywords")
                    print(f"[keywords] VPS keyword extraction (voice) for avatar {avatar_id}")
                except Exception as e:
                    print(f"[keywords] VPS extraction failed ({e}) — falling back to LLM mode")
                    keywords_by_lang, _web_q = generate_context_aware_keywords_for_multilingual_text_index_search(
                        text_query_llm, conversation_for_rag, rag_languages,
                        has_sensor=bool(avatar_sensor_tools.get(avatar_id)),
                    )
            else:
                keywords_by_lang, _web_q = generate_context_aware_keywords_for_multilingual_text_index_search(
                    text_query_llm, conversation_for_rag, rag_languages,
                    has_sensor=bool(avatar_sensor_tools.get(avatar_id)),
                )
            vtimings["keyword_gen_ms"] = round((time.perf_counter() - _tr) * 1000)
            _tr = time.perf_counter()
            rag_context = RAG(avatar_rag_tools[avatar_id], keywords_by_lang=keywords_by_lang)
            vtimings["rag_retrieval_ms"] = round((time.perf_counter() - _tr) * 1000)
            if rag_context:
                chat_history[-1]["content"] += f" <End of User message>. <<Context from knowledge base: {rag_context}>>"
                print("[Voice] RAG context injected (LLM keyword generation)")
                debug_log(avatar_id, "VOICE/RAG_CONTEXT", rag_context)
    except Exception as e:
        print(f"[Voice] RAG failed (continuing without context): {e}")

    # If this request triggered the avatar's RAG index cold load, surface it as
    # its own latency segment (otherwise it hides in "backend overhead").
    _load_ms = avatar_rag_tools.consume_load_ms(avatar_id)
    if _load_ms:
        vtimings["index_load_ms"] = _load_ms

    # Sensor tool prompt injection + always-fetch snapshot
    sensor_config = avatar_sensor_tools.get(avatar_id)
    if sensor_config:
        _ts = time.perf_counter()
        sensor_summary = _fetch_sensor_summary(avatar_id)
        vtimings["sensor_snapshot_ms"] = round((time.perf_counter() - _ts) * 1000)
        if sensor_summary:
            chat_history[0]["content"] += SENSOR_SNAPSHOT_INSTRUCTION.format(summary=sensor_summary)
        else:
            chat_history[0]["content"] += (
                "\n\nIf the user asks about live sensor data (temperature, pH, water quality etc.), "
                "say you'll look it up and call: analyze_sensor_data(user_query=\"...\"). "
                "Otherwise respond normally."
            )

    # Web search (model-invoked tool) — governed by the same per-avatar flag as
    # the text pipeline (llmDefaults.webSearchEnabled, toggled from the main
    # AvatarGarden page). When enabled, the model may answer with
    # web_search(query=...); the backend runs Brave and re-invokes with the
    # results labeled as an external source.
    avatars_cfg = json.load(open(avatars_path, "r"))
    avatar_cfg = next((a for a in avatars_cfg if str(a.get("id")) == avatar_id), None)
    web_search_enabled = bool((avatar_cfg or {}).get("llmDefaults", {}).get("webSearchEnabled", True))
    if web_search_enabled:
        chat_history[0]["content"] += VOICE_WEB_SEARCH_SCHEMA

    # Main LLM call
    llm = create_llm_instance(chat_provider, chat_model, system_prompt,
                              temperature=temperature, top_k=top_k, top_p=top_p,
                              task_name="voice_chat", providers=providers)
    _tllmv = time.perf_counter()
    response_id = str(_uuid.uuid4())

    _sse_first = {"sent": False}

    def _sse(delta=None, finish=False):
        # OpenAI streaming spec: role appears only in the first delta,
        # subsequent chunks carry content only.
        if delta is not None:
            d = {"content": delta} if _sse_first["sent"] else {"role": "assistant", "content": delta}
            _sse_first["sent"] = True
        else:
            d = {}
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": chat_model,
            "choices": [{"index": 0, "delta": d,
                         "finish_reason": "stop" if finish else None}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def _finalize(full_text):
        """Record timings and debug logs once the full response is known."""
        vtimings["total_backend_ms"] = round((time.perf_counter() - _vt0) * 1000)
        vtimings["ts"] = time.time()
        _last_voice_timings[avatar_id] = vtimings
        print(f"[TIMING] VOICE backend: {vtimings}")
        print(f"[Voice] Response: {full_text[:120]}...")
        debug_log(avatar_id, "VOICE/LLM_RESPONSE", full_text)
        debug_log(avatar_id, "VOICE/FULL_PROMPT", json.dumps(chat_history, ensure_ascii=False, indent=2))

    def _buffered_response():
        """Original non-streaming flow — also the fallback when streaming fails."""
        text = llm.complete(chat_history).text
        vtimings["main_llm_ms"] = round((time.perf_counter() - _tllmv) * 1000)
        _tst = time.perf_counter()
        text, sensor_handled = handle_sensor_tool_call(
            text, avatar_id, sensor_provider, sensor_model, llm, chat_history, providers=providers)
        if sensor_handled:
            vtimings["sensor_tool_ms"] = round((time.perf_counter() - _tst) * 1000)
            print("[Voice] Sensor analysis complete")
        if web_search_enabled and 'web_search(query="' in text:
            _tw = time.perf_counter()
            text, ws_handled = handle_web_search_tool_call(text, avatar_id, llm, chat_history)
            if ws_handled:
                vtimings["web_search_ms"] = round((time.perf_counter() - _tw) * 1000)
                print("[Voice] Web search complete")
        return sanitize_for_tts(text)

    def generate_buffered():
        text = _buffered_response()
        _finalize(text)
        yield _sse(text)
        yield _sse(finish=True)
        yield "data: [DONE]\n\n"

    def generate_streaming():
        """
        Stream sanitized sentences to Agora as the LLM produces them — TTS starts
        on the first sentence instead of waiting for the full response.
        The head of the stream is buffered briefly to detect sensor tool calls,
        which require the complete response (handled via the buffered flow).
        """
        buf, full_parts = "", []
        head_checked = False
        sensor_mode = False
        web_mode = False
        emitted = False
        try:
            for delta in llm.stream_deltas(chat_history):
                if "llm_first_token_ms" not in vtimings:
                    vtimings["llm_first_token_ms"] = round((time.perf_counter() - _tllmv) * 1000)
                full_parts.append(delta)
                buf += delta

                if not head_checked:
                    if len(buf) < 48 and "\n" not in buf:
                        continue
                    head_checked = True
                    if "analyze_sensor_data" in buf:
                        sensor_mode = True  # accumulate silently; tool needs the full call
                    elif web_search_enabled and 'web_search(query="' in buf:
                        web_mode = True  # accumulate silently; search + re-invoke at the end
                if sensor_mode or web_mode:
                    continue

                # Flush complete sentences (Agora starts TTS per chunk)
                while True:
                    m = _SENTENCE_END_RE.search(buf)
                    if not m:
                        if len(buf) > 300:  # long clause without boundary — flush anyway
                            clean = sanitize_for_tts(buf)
                            buf = ""
                            if clean:
                                emitted = True
                                yield _sse(clean + " ")
                        break
                    sent, buf = buf[:m.end()], buf[m.end():]
                    clean = sanitize_for_tts(sent)
                    if clean:
                        emitted = True
                        yield _sse(clean + " ")

            response_text = "".join(full_parts)
            vtimings["main_llm_ms"] = round((time.perf_counter() - _tllmv) * 1000)

            if sensor_mode or ("analyze_sensor_data" in response_text and not emitted):
                _tst = time.perf_counter()
                response_text, handled = handle_sensor_tool_call(
                    response_text, avatar_id, sensor_provider, sensor_model, llm, chat_history, providers=providers)
                if handled:
                    vtimings["sensor_tool_ms"] = round((time.perf_counter() - _tst) * 1000)
                    print("[Voice] Sensor analysis complete")
                response_text = sanitize_for_tts(response_text)
                yield _sse(response_text)
            elif web_mode or ('web_search(query="' in response_text and not emitted):
                # Web search tool call — silent accumulation completed; run Brave
                # and re-invoke, then emit the final response as one chunk.
                _tw = time.perf_counter()
                response_text, ws_handled = handle_web_search_tool_call(
                    response_text, avatar_id, llm, chat_history)
                if ws_handled:
                    vtimings["web_search_ms"] = round((time.perf_counter() - _tw) * 1000)
                    print("[Voice] Web search complete")
                response_text = sanitize_for_tts(response_text)
                yield _sse(response_text)
            else:
                tail = sanitize_for_tts(buf)
                if tail:
                    yield _sse(tail)
                response_text = sanitize_for_tts(response_text)

            yield _sse(finish=True)
            yield "data: [DONE]\n\n"
            _finalize(response_text)

        except Exception as e:
            print(f"[Voice] Streaming failed ({e}) — falling back to buffered response")
            if emitted:
                # Partial audio already went out; close the stream cleanly
                yield _sse(finish=True)
                yield "data: [DONE]\n\n"
                _finalize("".join(full_parts) + " [stream aborted]")
            else:
                yield from generate_buffered()

    # Streaming is opt-in per session via the agent's llm.url (?streaming=1),
    # set by the voice lab toggle — off by default while Agora-side chunk
    # handling is being ironed out (see ISSUES.md 14). Buffered remains the
    # default and the automatic fallback on stream errors.
    use_streaming = request.args.get("streaming") == "1"
    gen = generate_streaming() if use_streaming else generate_buffered()
    return Response(
        stream_with_context(gen),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.register_blueprint(voice_bp)


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, port=5001)
