import { Link } from 'react-router-dom';
import { useEffect, useRef, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";
import LatencyBreakdown from "./LatencyBreakdown";

export default function MultiAvatarChat() {
  // const [refreshPromptState, setRefreshPromptState] = useState("idle");
  // const [refreshEmbeddingsState, setRefreshEmbeddingsState] = useState("idle");

  // --- avatar management state ---
  const [avatars, setAvatars] = useState([]);
  const [selectedAvatarId, setSelectedAvatarId] = useState("");
  const [avatarFormOpen, setAvatarFormOpen] = useState(false);
  const [avatarFormMode, setAvatarFormMode] = useState("create"); // "create" | "edit"
  const [avatarForm, setAvatarForm] = useState({
    id: null,
    name: "",
    systemPromptUrl: "",
    contextDocsUrl: "",
    sensorApiUrl: "",
    sensorDescription: "",
    ragLanguages: "en, de",
  });
  const [avatarSaving, setAvatarSaving] = useState(false);
  const [avatarError, setAvatarError] = useState("");

  // --- chat state (same as original, just reused) ---
  const [defaultMessages, setDefaultMessages] = useState([]);
  const [debateMessages, setDebateMessages] = useState([]);
  const [input, setInput] = useState("");
  const [defaultThinking, setDefaultThinking] = useState(false);
  const [debateThinking, setDebateThinking] = useState(false);
  const [isDebateMode, setIsDebateMode] = useState(false);

  // --- LLM selection and health state ---
  // Separate providers for each task
  const [currentUserChatProvider, setCurrentUserChatProvider] = useState('gwdg');
  const [currentUserChatModel, setCurrentUserChatModel] = useState('gwdg/gemma-3-27b-it');
  const [currentUserTextQueryProvider, setCurrentUserTextQueryProvider] = useState('gwdg');
  const [currentUserTextQueryModel, setCurrentUserTextQueryModel] = useState('gwdg/gemma-3-27b-it');
  const [currentUserSensorProvider, setCurrentUserSensorProvider] = useState('gwdg');
  const [currentUserSensorModel, setCurrentUserSensorModel] = useState('mistral-large-instruct');
  const [currentUserVoiceChatProvider, setCurrentUserVoiceChatProvider] = useState('gwdg');
  const [currentUserVoiceChatModel, setCurrentUserVoiceChatModel] = useState('gwdg/qwen3-30b-a3b-instruct-2507');
  const [currentUserTemperature, setCurrentUserTemperature] = useState(0.7);
  const [currentUserTopK, setCurrentUserTopK] = useState(40);
  const [currentUserTopP, setCurrentUserTopP] = useState(1.0);
  const [modelHealth, setModelHealth] = useState({});
  const [llmOptions, setLlmOptions] = useState({});
  const [llmOptionsLastRefreshed, setLlmOptionsLastRefreshed] = useState(null);
  const [llmConfigExpanded, setLlmConfigExpanded] = useState(false);
  const [webSearchOn, setWebSearchOn] = useState(null);
  const [hasLlmDefaults, setHasLlmDefaults] = useState(false);
  const [adminDefaultModels, setAdminDefaultModels] = useState({ chat: null, textQuery: null, sensor: null, voiceChat: null });

  // --- backend log viewer state ---
  const [logExpanded, setLogExpanded] = useState(false);
  const [backendLog, setBackendLog] = useState("");

  // --- latency analysis state ---
  const [latencyExpanded, setLatencyExpanded] = useState(false);
  const [lastLatency, setLastLatency] = useState(null); // { roundTripMs, timings }

  // --- TTS voice config state (Cartesia, per-avatar) ---
  const [ttsVoices, setTtsVoices] = useState([]);
  const [ttsVoiceSel, setTtsVoiceSel] = useState("");
  const [ttsLangSel, setTtsLangSel] = useState("");
  const [ttsSaving, setTtsSaving] = useState(false);
  const [ttsMsg, setTtsMsg] = useState("");
  const [ragPinSaving, setRagPinSaving] = useState(false);
  const backendLogRef = useRef(null);
  const backendLogPreRef = useRef(null);
  const backendLogFirstLoadRef = useRef(true);

  // --- Integrations state ---
  const [integrationsExpanded, setIntegrationsExpanded] = useState(false);
  const [integrations, setIntegrations] = useState({});
  const [integrationEdits, setIntegrationEdits] = useState({});
  const [integrationSaving, setIntegrationSaving] = useState(false);
  const [integrationMsg, setIntegrationMsg] = useState("");

  // --- LLM provider key editing state ---
  const [llmProvidersList, setLlmProvidersList] = useState([]);
  const [providerKeyEdits, setProviderKeyEdits] = useState({});
  const [providerKeySaving, setProviderKeySaving] = useState({});

  // --- LLM provider management state ---
  const [providerFormOpen, setProviderFormOpen] = useState(false);
  const [providerSaving, setProviderSaving] = useState(false);
  const [providerError, setProviderError] = useState("");
  const [providerForm, setProviderForm] = useState({
    id: "",
    name: "",
    provider_key: "custom",
    api_base: "",
    api_key: "",
    models: "",
  });

  // === Initial load ===
  useEffect(() => {
    // Fetch avatars
    (async () => {
      try {
        const resp = await fetch("/api/avatars");
        if (!resp.ok) throw new Error("Failed to fetch avatars");
        const data = await resp.json();
        setAvatars(data || []);
        if (data && data.length > 0) {
          setSelectedAvatarId(data[0].id);
        }
      } catch (err) {
        console.error("Error loading avatars:", err);
      }
    })();

    // Fetch LLM options
    (async () => {
      try {
        const resp = await fetch("/api/llm-options");
        if (!resp.ok) throw new Error("Failed to fetch LLM options");
        const data = await resp.json();
        setLlmOptions(data.options);
        setLlmOptionsLastRefreshed(data.last_refreshed);
        // Set initial model if available
        const firstProvider = Object.keys(data.options)[0];
        if (firstProvider && data.options[firstProvider].models.length > 0) {
          setCurrentUserChatProvider(firstProvider);
          setCurrentUserChatModel(data.options[firstProvider].models[0]);
          setCurrentUserTextQueryProvider(firstProvider);
          setCurrentUserTextQueryModel(data.options[firstProvider].models[0]);
          setCurrentUserSensorProvider(firstProvider);
          setCurrentUserSensorModel(data.options[firstProvider].models[0]);
        }
      } catch (err) {
        console.error("Error loading LLM options:", err);
      }
    })();

    // Fetch integrations
    fetch("/api/voice/tts-voices")
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setTtsVoices(data); })
      .catch(e => console.error("Failed to load TTS voices:", e));
    fetch("/api/integrations")
      .then(r => r.json())
      .then(setIntegrations)
      .catch(console.error);

    // Fetch LLM providers list (includes key_set / key_last4)
    fetch("/api/llm-providers")
      .then(r => r.json())
      .then(setLlmProvidersList)
      .catch(console.error);
  }, []);

  // === Background health check (non-blocking) ===
  const healthCheckInProgress = useRef(false);

  useEffect(() => {
    // Skip if a health check is already in progress (prevents StrictMode double-invocation)
    if (healthCheckInProgress.current) {
      console.log("Health check already in progress, skipping duplicate");
      return;
    }
    healthCheckInProgress.current = true;

    const fetchHealthStatus = async () => {
      // Fetch fast and slow endpoints in parallel
      const fetchFast = async () => {
        try {
          const resp = await fetch("/api/health/llm/fast");
          if (!resp.ok) throw new Error("Failed to fetch fast health");
          const data = await resp.json();
          setModelHealth(prev => ({ ...prev, ...data }));
          console.log("Fast LLM health loaded:", data);
        } catch (err) {
          console.error("Error loading fast LLM health:", err);
        }
      };

      const fetchSlow = async () => {
        try {
          const resp = await fetch("/api/health/llm/slow");
          if (!resp.ok) throw new Error("Failed to fetch slow health");
          const data = await resp.json();
          setModelHealth(prev => ({ ...prev, ...data }));
          console.log("Slow LLM health loaded:", data);
        } catch (err) {
          console.error("Error loading slow LLM health:", err);
        }
      };

      const fetchOpenRouter = async () => {
        try {
          const resp = await fetch("/api/health/llm/openrouter");
          if (!resp.ok) throw new Error("Failed to fetch OpenRouter health");
          const data = await resp.json();
          setModelHealth(prev => ({ ...prev, ...data }));
          console.log("OpenRouter LLM health loaded:", data);
        } catch (err) {
          console.error("Error loading OpenRouter LLM health:", err);
        }
      };

      // All three run in parallel - each updates UI incrementally as it completes
      fetchFast();
      fetchSlow();
      fetchOpenRouter();
    };

    // Fetch health status in background after component mounts
    fetchHealthStatus();
  }, []);

  // === Backend log polling ===
  const fetchBackendLog = async () => {
    try {
      const resp = await fetch("/api/backend-log");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setBackendLog(data.log);
    } catch (e) {
      console.error("Error loading backend log:", e);
    }
  };

  useEffect(() => {
    if (!logExpanded) return;
    fetchBackendLog();
    const timer = setInterval(fetchBackendLog, 3000);
    return () => clearInterval(timer);
  }, [logExpanded]);

  useEffect(() => {
    if (backendLog.length === 0) return;
    if (backendLogFirstLoadRef.current) {
      backendLogFirstLoadRef.current = false;
      backendLogRef.current?.scrollIntoView();
      return;
    }
    const el = backendLogPreRef.current;
    if (!el) return;
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50;
    if (isNearBottom) {
      backendLogRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [backendLog]);

  // === Load Admin defaults when avatar changes ===
  useEffect(() => {
    if (!selectedAvatarId || !llmOptions || Object.keys(llmOptions).length === 0) return;

    const selectedAvatar = avatars.find(a => a.id === selectedAvatarId);
    const firstProvider = Object.keys(llmOptions)[0];
    const resolveProvider = (p) => (p && llmOptions[p]) ? p : firstProvider;

    // Returns { provider, model }. When the saved model can't be used (hidden
    // provider, model no longer listed), fall back to a model that is not
    // marked offline — a blind models[0] fallback once landed on an offline
    // model while the dropdown displayed a different, online one.
    const resolveModel = (p, m) => {
      const provider = resolveProvider(p);
      const models = llmOptions[provider]?.models || [];
      if (m && models.includes(m)) return { provider, model: m };
      const online = models.find(mod => modelHealth[mod] !== 'offline');
      return { provider, model: online || models[0] || '' };
    };

    if (selectedAvatar?.llmDefaults) {
      const defaults = selectedAvatar.llmDefaults;
      console.log("Loading Admin defaults for avatar:", defaults);

      // Load chat defaults
      if (defaults.chat) {
        const { provider: cp, model: cm } = resolveModel(defaults.chat.provider, defaults.chat.model);
        setCurrentUserChatProvider(cp);
        setCurrentUserChatModel(cm);
        if (defaults.chat.temperature) setCurrentUserTemperature(defaults.chat.temperature);
        if (defaults.chat.top_k) setCurrentUserTopK(defaults.chat.top_k);
        if (defaults.chat.top_p) setCurrentUserTopP(defaults.chat.top_p);
      }

      // Load text query defaults
      if (defaults.textQuery) {
        const { provider: tp, model: tm } = resolveModel(defaults.textQuery.provider, defaults.textQuery.model);
        setCurrentUserTextQueryProvider(tp);
        setCurrentUserTextQueryModel(tm);
      }

      // Load sensor defaults
      if (defaults.sensor) {
        const { provider: sp, model: sm } = resolveModel(defaults.sensor.provider, defaults.sensor.model);
        setCurrentUserSensorProvider(sp);
        setCurrentUserSensorModel(sm);
      }

      // Load voice chat defaults (falls back to chat when unset — mirrors backend)
      if (defaults.voiceChat) {
        const { provider: vp, model: vm } = resolveModel(defaults.voiceChat.provider, defaults.voiceChat.model);
        setCurrentUserVoiceChatProvider(vp);
        setCurrentUserVoiceChatModel(vm);
      } else if (defaults.chat) {
        const { provider: vp, model: vm } = resolveModel(defaults.chat.provider, defaults.chat.model);
        setCurrentUserVoiceChatProvider(vp);
        setCurrentUserVoiceChatModel(vm);
      }

      setAdminDefaultModels({
        chat: defaults.chat?.model || null,
        textQuery: defaults.textQuery?.model || null,
        sensor: defaults.sensor?.model || null,
        voiceChat: defaults.voiceChat?.model || null,
      });
      setHasLlmDefaults(true);
    } else {
      setAdminDefaultModels({ chat: null, textQuery: null, sensor: null, voiceChat: null });
      // No Admin defaults saved — fall back to the first visible provider's
      // first non-offline model
      if (firstProvider && llmOptions[firstProvider].models.length > 0) {
        const { model: fallbackModel } = resolveModel(firstProvider, null);
        setCurrentUserChatProvider(firstProvider);
        setCurrentUserChatModel(fallbackModel);
        setCurrentUserTextQueryProvider(firstProvider);
        setCurrentUserTextQueryModel(fallbackModel);
        setCurrentUserSensorProvider(firstProvider);
        setCurrentUserSensorModel(fallbackModel);
      }
      setCurrentUserTemperature(0.7);
      setCurrentUserTopK(40);
      setCurrentUserTopP(1.0);
      setHasLlmDefaults(false);
    }
  }, [selectedAvatarId, llmOptions]);

  // === Save/Clear Admin defaults ===
  const handleSaveLlmDefaults = async () => {
    if (!selectedAvatarId) return;

    try {
      const resp = await fetch(`/api/avatars/${selectedAvatarId}/llm-defaults`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chatProvider: currentUserChatProvider,
          chatModel: currentUserChatModel,
          temperature: currentUserTemperature,
          topK: currentUserTopK,
          topP: currentUserTopP,
          textQueryProvider: currentUserTextQueryProvider,
          textQueryModel: currentUserTextQueryModel,
          sensorProvider: currentUserSensorProvider,
          sensorModel: currentUserSensorModel,
          voiceChatProvider: currentUserVoiceChatProvider,
          voiceChatModel: currentUserVoiceChatModel,
        }),
      });

      if (!resp.ok) throw new Error("Failed to save defaults");
      const updatedAvatar = await resp.json();

      // Update avatars list with the new data
      setAvatars(prev => prev.map(a => a.id === selectedAvatarId ? updatedAvatar : a));
      setHasLlmDefaults(true);
      console.log("Admin defaults saved for avatar:", selectedAvatarId);
    } catch (err) {
      console.error("Error saving Admin defaults:", err);
    }
  };

  const handleClearLlmDefaults = async () => {
    if (!selectedAvatarId) return;

    try {
      const resp = await fetch(`/api/avatars/${selectedAvatarId}/llm-defaults`, {
        method: 'DELETE',
      });

      if (!resp.ok) throw new Error("Failed to clear defaults");
      const updatedAvatar = await resp.json();

      // Update avatars list
      setAvatars(prev => prev.map(a => a.id === selectedAvatarId ? updatedAvatar : a));
      setHasLlmDefaults(false);
      console.log("Admin defaults cleared for avatar:", selectedAvatarId);
    } catch (err) {
      console.error("Error clearing Admin defaults:", err);
    }
  };

  // === API Keys save handler (integrations + LLM provider keys) ===
  const handleSaveIntegrations = async () => {
    setIntegrationSaving(true);
    setIntegrationMsg("");
    try {
      const saved = [];

      // Save integration keys (Brave, Agora, Deepgram, Cartesia)
      if (Object.keys(integrationEdits).length > 0) {
        const resp = await fetch("/api/integrations", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(integrationEdits),
        });
        const result = await resp.json();
        if (result.updated?.length) saved.push(...result.updated);
        setIntegrationEdits({});
        const fresh = await fetch("/api/integrations").then(r => r.json());
        setIntegrations(fresh);
      }

      // Save LLM provider keys
      const providerUpdates = Object.entries(providerKeyEdits).filter(([, v]) => v);
      await Promise.all(providerUpdates.map(async ([providerId, key]) => {
        const resp = await fetch(`/api/llm-providers/${providerId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_key: key }),
        });
        if (resp.ok) saved.push(providerId);
      }));
      if (providerUpdates.length > 0) {
        setProviderKeyEdits({});
        const optResp = await fetch("/api/llm-providers");
        if (optResp.ok) setLlmProvidersList(await optResp.json());
      }

      setIntegrationMsg(saved.length ? `Saved: ${saved.join(", ")}` : "Nothing to save (fields were empty)");
    } catch (e) {
      setIntegrationMsg("Error saving keys");
    } finally {
      setIntegrationSaving(false);
    }
  };

  const [topics] = useState([
    'The River should have legal personhood',
    'The River should be able to own property',
    'There should exist a "River Fund”',
    'The Avatar should be able to legally speak on behalf of the River'
  ]);

  const topicDescriptions = {
    'The River should have legal personhood': "In recent years, rivers around the world have been granted legal personhood to recognize their intrinsic rights and protect their ecosystems. Granting the Lahn legal personhood would mean treating the river not merely as a resource but as a living entity with legal standing - analogous to the legal standing that a person or corporation holds. This shift could reshape how environmental protection is approached in the region, allowing for the river's interests to be formally represented in legal and political systems. And even create precedent for the river suing a company or the government, for example.",
    'The River should be able to own property': "If the Lahn were recognized as a legal person, it could theoretically hold property titles. This would allow the river to directly control land essential to its health—such as floodplains, wetlands, or riverbanks—ensuring its ecological integrity is not compromised by conflicting human interests. Property ownership could become a tool for the river to safeguard its own regeneration and future.",
    'There should exist a “River Fund”': "A dedicated “Lahn Fund” would serve as a financial mechanism to support the ongoing protection, restoration, and stewardship of the river. This fund could receive public and private contributions, fines from environmental damages, or a share of local economic activities that depend on the river. Managed in the river’s interest, the fund could finance ecological research, conservation projects, community engagement, and support the operational costs of the Avatar or legal guardianship system.",
    'The Avatar should be able to legally speak on behalf of the River': "The Lahn Avatar is envisioned as a voice for the river—an interface between natural and human systems. Allowing the Avatar to legally speak on behalf of the Lahn would formalize its role as a representative entity in decision-making processes. This could enable the river’s interests to be expressed in public hearings, governmental deliberations, and community forums, fostering a new model of ecological democracy and interspecies governance."
  };

  const [selectedTopic, setSelectedTopic] = useState("");
  const [debateSummary, setDebateSummary] = useState(`Avatar:\nPro:\nCon:\n\nYou:\nPro:\nCon:`);
  // const [hasFetchedDebateInit, setHasFetchedDebateInit] = useState(false);

  const chatEndRef = useRef(null);

  const [refreshState, setRefreshState] = useState({ prompt: "idle", embeddings: "idle" }); // idle, loading, success, error

  const handleRefresh = async (type) => {
    if (!selectedAvatarId) return;

    const endpoint = type === 'prompt' ? '/api/refresh-prompt' : '/api/refresh-embeddings';
    setRefreshState(prev => ({ ...prev, [type]: 'loading' }));

    try {
      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ avatar_id: selectedAvatarId }),
      });

      if (!resp.ok) {
        const errorData = await resp.json();
        throw new Error(errorData.error || `Failed to refresh ${type}`);
      }

      setRefreshState(prev => ({ ...prev, [type]: 'success' }));
      setTimeout(() => setRefreshState(prev => ({ ...prev, [type]: 'idle' })), 2000);

    } catch (err) {
      console.error(`Error refreshing ${type}:`, err);
      setRefreshState(prev => ({ ...prev, [type]: 'error' }));
      setTimeout(() => setRefreshState(prev => ({ ...prev, [type]: 'idle' })), 3000);
    }
  };

  const messages = isDebateMode ? debateMessages : defaultMessages;
  const setMessages = isDebateMode ? setDebateMessages : setDefaultMessages;
  const isThinking = isDebateMode ? debateThinking : defaultThinking;
  const setIsThinking = isDebateMode ? setDebateThinking : setDefaultThinking;

  const selectedAvatar = avatars.find(a => a.id === selectedAvatarId) || null;
  const keywordLlmOn = selectedAvatar ? selectedAvatar.keywordMode === "llm" : true;
  const vpsVoiceOn = selectedAvatar ? (selectedAvatar.llmDefaults?.voiceBackend === "vps") : false;

  const handleToggleVoiceBackend = async (checked) => {
    if (!selectedAvatarId) return;
    try {
      const resp = await fetch(`/api/avatars/${selectedAvatarId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ voiceBackend: checked ? "vps" : "convoai" }),
      });
      if (!resp.ok) throw new Error((await resp.json()).error || "Save failed");
      const updated = await fetch("/api/avatars").then(r => r.json());
      setAvatars(updated);
    } catch (e) {
      console.error("Failed to toggle voice backend:", e);
    }
  };

  const handleToggleKeywordMode = async (checked) => {
    if (!selectedAvatarId) return;
    try {
      const resp = await fetch(`/api/avatars/${selectedAvatarId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keywordMode: checked ? "llm" : "vps" }),
      });
      if (!resp.ok) throw new Error((await resp.json()).error || "Save failed");
      const updated = await fetch("/api/avatars").then(r => r.json());
      setAvatars(updated);
    } catch (e) {
      console.error("Failed to toggle keyword mode:", e);
    }
  };

  const firstRender = useRef(true);

  // === Avatar CRUD helpers ===

  const openCreateAvatarForm = () => {
    setAvatarFormMode("create");
    setAvatarForm({
      id: null,
      name: "",
      systemPromptUrl: "",
      contextDocsUrl: "",
      sensorApiUrl: "",
      sensorDescription: "",
    });
    setAvatarError("");
    setAvatarFormOpen(true);
  };

  const openEditAvatarForm = (avatar) => {
    setAvatarFormMode("edit");
    setAvatarForm({
      id: avatar.id,
      name: avatar.name || "",
      systemPromptUrl: avatar.systemPromptUrl || "",
      contextDocsUrl: avatar.contextDocsUrl || "",
      sensorApiUrl: avatar.sensorApiUrl || "",
      sensorDescription: avatar.sensorDescription || "",
      ragLanguages: Array.isArray(avatar.ragLanguages)
        ? avatar.ragLanguages.join(", ")
        : (avatar.ragLanguages || "en, de"),
    });
    setAvatarError("");
    setAvatarFormOpen(true);
  };

  const handleSetDefaultAvatar = async () => {
    if (!selectedAvatarId) return;
    try {
      const resp = await fetch(`/api/avatars/${selectedAvatarId}/set-default`, {
        method: "POST",
      });
      if (!resp.ok) throw new Error("Failed to set default avatar");
      // Backend returns the reordered list — pinned default first. Page-load
      // selection follows data[0], so the pinned avatar survives refreshes.
      setAvatars(await resp.json());
    } catch (e) {
      console.error("Failed to set default avatar:", e);
    }
  };

  const handleAvatarFormChange = (field, value) => {
    setAvatarForm(prev => ({ ...prev, [field]: value }));
  };

  const handleAvatarFormSubmit = async (e) => {
    e.preventDefault();
    setAvatarError("");

    if (!avatarForm.name.trim()) {
      setAvatarError("Avatar name is required.");
      return;
    }

    setAvatarSaving(true);
    try {
      if (avatarFormMode === "create") {
        const resp = await fetch("/api/avatars", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: avatarForm.name,
            systemPromptUrl: avatarForm.systemPromptUrl,
            contextDocsUrl: avatarForm.contextDocsUrl,
            sensorApiUrl: avatarForm.sensorApiUrl,
            sensorDescription: avatarForm.sensorDescription,
            ragLanguages: avatarForm.ragLanguages,
          }),
        });
        if (!resp.ok) throw new Error("Failed to create avatar");
        const newAvatar = await resp.json(); // expect { id, name, ... }
        setAvatars(prev => [...prev, newAvatar]);
        setSelectedAvatarId(newAvatar.id);
      } else {
        const resp = await fetch(`/api/avatars/${avatarForm.id}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: avatarForm.name,
            systemPromptUrl: avatarForm.systemPromptUrl,
            contextDocsUrl: avatarForm.contextDocsUrl,
            sensorApiUrl: avatarForm.sensorApiUrl,
            sensorDescription: avatarForm.sensorDescription,
            ragLanguages: avatarForm.ragLanguages,
          }),
        });
        if (!resp.ok) throw new Error("Failed to update avatar");
        const updated = await resp.json();
        setAvatars(prev =>
          prev.map(a => (a.id === updated.id ? updated : a))
        );
      }

      setAvatarFormOpen(false);
    } catch (err) {
      console.error(err);
      setAvatarError("Could not save avatar. Please try again.");
    } finally {
      setAvatarSaving(false);
    }
  };

  // === LLM Provider Management helpers ===

  const openCreateProviderForm = () => {
    setProviderForm({
      id: "",
      name: "",
      provider_key: "custom",
      api_base: "",
      api_key: "",
      models: "",
    });
    setProviderError("");
    setProviderFormOpen(true);
  };

  const handleProviderFormChange = (field, value) => {
    setProviderForm(prev => ({ ...prev, [field]: value }));
  };

  const handleProviderFormSubmit = async (e) => {
    e.preventDefault();
    setProviderError("");

    if (!providerForm.name.trim()) {
      setProviderError("Provider name is required.");
      return;
    }

    if (!providerForm.models.trim()) {
      setProviderError("At least one model is required (comma-separated).");
      return;
    }

    setProviderSaving(true);
    try {
      const modelsList = providerForm.models.split(",").map(m => m.trim()).filter(m => m);

      const resp = await fetch("/api/llm-providers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: providerForm.id || undefined,
          name: providerForm.name,
          provider_key: providerForm.provider_key,
          api_base: providerForm.api_base,
          api_key: providerForm.api_key,
          models: modelsList,
        }),
      });

      if (!resp.ok) {
        const errData = await resp.json();
        throw new Error(errData.error || "Failed to create provider");
      }

      const newProvider = await resp.json();

      // Refresh LLM options from server
      const optionsResp = await fetch("/api/llm-options");
      if (optionsResp.ok) {
        const newOptions = await optionsResp.json();
        setLlmOptions(newOptions.options);
        setLlmOptionsLastRefreshed(newOptions.last_refreshed);
      }

      // Refresh health check to get status for new provider's models
      const healthResp = await fetch("/api/health/llm");
      if (healthResp.ok) {
        const healthData = await healthResp.json();
        setModelHealth(healthData);
      }

      setProviderFormOpen(false);

      // Select the newly created provider and its first model
      setCurrentUserChatProvider(newProvider.id);
      setCurrentUserTextQueryProvider(newProvider.id);
      setCurrentUserSensorProvider(newProvider.id);
      if (newProvider.models && newProvider.models.length > 0) {
        setCurrentUserChatModel(newProvider.models[0]);
        setCurrentUserTextQueryModel(newProvider.models[0]);
        setCurrentUserSensorModel(newProvider.models[0]);
      }
    } catch (err) {
      console.error(err);
      setProviderError(err.message || "Could not save provider. Please try again.");
    } finally {
      setProviderSaving(false);
    }
  };


  // === Chat fetch ===
  const fetchMessage = async (payload) => {
    if (!selectedAvatar) return;

    console.log("fetchMessage called with prompt:", payload.prompt, "history:", payload.history, "avatar:", selectedAvatar);

    setIsThinking(true);
    const t0 = performance.now();
    try {
      const resp = await fetch(
        "/api/chat?avatar="+selectedAvatar.id,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...payload,
            avatarId: selectedAvatar.id,
            avatarName: selectedAvatar.name,
            chatProvider: currentUserChatProvider,
            chatModel: currentUserChatModel,
            temperature: currentUserTemperature,
            topK: currentUserTopK,
            topP: currentUserTopP,
            textQueryProvider: currentUserTextQueryProvider,
            textQueryModel: currentUserTextQueryModel,
            sensorProvider: currentUserSensorProvider,
            sensorModel: currentUserSensorModel,
            ...(isDebateMode && selectedTopic
              ? { topic: selectedTopic }
              : {})
          }),
        }
      );
      const data = await resp.json();
      const roundTripMs = Math.round(performance.now() - t0);
      if (data.error) {
        setMessages(prev => [...prev, { sender: "avatar", text: `⚠️ ${data.error}` }]);
      } else {
        setMessages(prev => [...prev, { sender: "avatar", text: data.reply }]);
        if (data.timings) setLastLatency({ roundTripMs, timings: data.timings });
      }
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { sender: "avatar", text: "⚠️ An unexpected error occurred. Please try again." }]);
    } finally {
      setIsThinking(false);
    }
  };

  // === Debate summary effect ===
  useEffect(() => {
    const last = debateMessages[debateMessages.length - 1];
    if (isDebateMode && selectedTopic && last?.sender === 'avatar' && selectedAvatar) {
      (async () => {
        try {
          const resp = await fetch(
            "/api/debate-summary",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                history: debateMessages,
                topic: selectedTopic,
                summary: debateSummary,
                avatarId: selectedAvatar.id,
                avatarName: selectedAvatar.name,
              }),
            }
          );
          const { summary } = await resp.json();

          const formatted = summary
            .replace(/\nYou:/, '\n<hr/>\nYou:')
            .replace(/\b(Avatar|You|Pro|Con)\b/g, '<b>$1</b>');

          setDebateSummary(formatted);
        } catch (error) {
          console.error(error);
        }
      })();
    }
  }, [debateMessages, isDebateMode, selectedTopic, selectedAvatar, debateSummary]);

  const handleSubmit = async () => {
    if (!input.trim() || !selectedAvatar) return;
    const userInput = input;
    const updated = [...messages, { sender: "user", text: userInput }];
    setMessages(updated);
    setInput("");
    setIsThinking(true);
    await fetchMessage({ history: updated, prompt: userInput });
  };

  useEffect(() => {
    // Sync TTS voice selection with the selected avatar's stored config
    const av = avatars.find(a => a.id === selectedAvatarId);
    setTtsVoiceSel(av?.ttsVoiceId || "");
    setTtsLangSel(av?.ttsLanguage || "");
    setTtsMsg("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAvatarId, avatars]);

  const handleSaveTtsConfig = async () => {
    if (!selectedAvatarId) return;
    setTtsSaving(true);
    setTtsMsg("");
    try {
      const resp = await fetch(`/api/avatars/${selectedAvatarId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ttsVoiceId: ttsVoiceSel, ttsLanguage: ttsLangSel }),
      });
      if (!resp.ok) throw new Error((await resp.json()).error || "Save failed");
      const updated = await fetch("/api/avatars").then(r => r.json());
      setAvatars(updated);
      setTtsMsg("Saved — applies to new voice sessions");
    } catch (e) {
      setTtsMsg(`Error: ${e.message}`);
    } finally {
      setTtsSaving(false);
    }
  };

  const handleToggleRagPinned = async (checked) => {
    if (!selectedAvatarId) return;
    setRagPinSaving(true);
    try {
      const resp = await fetch(`/api/avatars/${selectedAvatarId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ragPinned: checked }),
      });
      if (!resp.ok) throw new Error((await resp.json()).error || "Save failed");
      const updated = await fetch("/api/avatars").then(r => r.json());
      setAvatars(updated);
    } catch (e) {
      console.error("Failed to toggle RAG pinning:", e);
    } finally {
      setRagPinSaving(false);
    }
  };

  useEffect(() => {
    // Reset chat when switching avatars
    setDefaultMessages([]);
    setDebateMessages([]);
    setInput("");

    setDefaultThinking(false);
    setDebateThinking(false);

    // Optional: reset debate UI too
    setSelectedTopic("");
    setDebateSummary(`Avatar:\nPro:\nCon:\n\nYou:\nPro:\nCon:`);

    // Optional: prevent auto-scroll jump on this reset
    firstRender.current = true;
  }, [selectedAvatarId]);

  // === Web search state is per-avatar — refetch whenever the selection changes ===
  useEffect(() => {
    if (!selectedAvatarId) return;
    let cancelled = false;
    setWebSearchOn(null);
    fetch(`/api/voice/web-search-status?avatar=${selectedAvatarId}`)
      .then(r => r.json())
      .then(d => { if (!cancelled) setWebSearchOn(d.webSearchEnabled); })
      .catch(() => { if (!cancelled) setWebSearchOn(null); });
    return () => { cancelled = true; };
  }, [selectedAvatarId]);


  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return; // skip scrolling on initial load
    }
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const offlineBlockers = [
    { role: 'chat', model: currentUserChatModel },
    { role: 'text query', model: currentUserTextQueryModel },
    { role: 'sensor', model: currentUserSensorModel },
  ].filter(({ model }) => modelHealth[model] === 'offline');

  const chatDisabled = !selectedAvatar || offlineBlockers.length > 0;

  return (
    <div
      className="min-h-screen p-4 flex flex-col items-center"
    >
      <motion.h1
        className="text-2xl md:text-3xl font-poetic font-semibold tracking-tight text-garden-ink mb-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        Avatar Garden
      </motion.h1>

      <Link
        to="/avatars-chat-voice"
        className="mb-4 inline-flex items-center gap-2 rounded-md bg-garden-water px-5 py-2 text-sm font-medium text-garden-paper hover:bg-garden-water/80 transition-colors"
      >
        🎙 Talk by voice
      </Link>

      {/* Avatar and LLM selection + controls */}
      <div className="w-full max-w-5xl flex flex-col gap-4 mb-4 px-4">
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
          <div className="flex-1">
            <label className="block mb-1 font-poetic text-garden-ink">
              Active avatar
            </label>
            <div className="flex flex-col sm:flex-row gap-2 items-stretch sm:items-center">
              <select
                className="flex-1 p-2 rounded-md border bg-garden-paper font-poetic"
                value={selectedAvatarId}
                onChange={e => setSelectedAvatarId(e.target.value)}
              >
                <option value="">-- Select an avatar --</option>
                {avatars.map(av => (
                  <option key={av.id} value={av.id}>
                    {av.name}
                  </option>
                ))}
              </select>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  className="font-poetic flex-1 sm:flex-none"
                  onClick={openCreateAvatarForm}
                >
                  + New
                </Button>
                {selectedAvatar && (
                  <Button
                    variant="outline"
                    className="font-poetic flex-1 sm:flex-none"
                    onClick={handleSetDefaultAvatar}
                  >
                    {selectedAvatar.defaultAvatar ? "Default ✓" : "Pin as default"}
                  </Button>
                )}
                {selectedAvatar && (
                  <Button
                    variant="outline"
                    className="font-poetic flex-1 sm:flex-none"
                    onClick={() => openEditAvatarForm(selectedAvatar)}
                  >
                    Edit
                  </Button>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Switch checked={isDebateMode} onCheckedChange={setIsDebateMode} />
              <span className="font-poetic text-garden-inksoft">Debate Mode</span>
            </div>
          </div>
        </div>

        {/* Horizontal separator */}
        <hr className="w-full border-garden-line border-dashed" />

        {/* Collapsible LLM Config Section */}
        <div className="w-full">
          <button
            className="flex items-center gap-2 font-poetic text-garden-inksoft cursor-pointer hover:text-garden-ink"
            onClick={() => setLlmConfigExpanded(!llmConfigExpanded)}
          >
            <span className="text-lg">{llmConfigExpanded ? '▼' : '▶'}</span>
            <span className="font-semibold">LLM Config</span>
          </button>

          {llmConfigExpanded && (
            <div className="mt-2 space-y-3">
              {/* Web search toggle */}
              <div className="flex items-center justify-between p-2 rounded-lg border bg-garden-paper2">
                <span className="font-poetic text-sm text-garden-inksoft">Web search (internet access)</span>
                <button
                  className={"relative w-12 h-6 rounded-full transition-colors duration-200 " + (webSearchOn ? "bg-garden-moss" : "bg-garden-line")}
                  onClick={async () => {
                    const newState = !webSearchOn;
                    setWebSearchOn(newState);
                    try {
                      await fetch("/api/voice/web-search-toggle", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ avatar_id: selectedAvatarId, enabled: newState }),
                      });
                    } catch (e) { setWebSearchOn(!newState); }
                  }}
                  disabled={webSearchOn === null}
                >
                  <span className={"absolute top-0.5 left-0.5 w-5 h-5 bg-garden-paper rounded-full shadow transition-transform duration-200 " + (webSearchOn ? "translate-x-6" : "translate-x-0")} />
                </button>
              </div>
              {/* ── Text ── */}
              <p className="font-poetic text-garden-inksoft text-xs font-semibold uppercase tracking-wider pt-1">Text</p>

              {/* Text chat provider and model */}
              <div className="p-3 rounded-lg border bg-garden-paper2 space-y-2">
                  <label className="block font-poetic text-garden-ink font-semibold text-sm">Text Chat</label>
                  <div className="flex flex-wrap items-center gap-2">
                      <select
                          className="flex-1 min-w-[120px] p-2 rounded-md border bg-garden-paper font-poetic text-sm"
                          value={currentUserChatProvider}
                          onChange={e => {
                              setCurrentUserChatProvider(e.target.value);
                              setCurrentUserChatModel(llmOptions[e.target.value].models[0]);
                          }}
                      >
                          {Object.keys(llmOptions).map(key => (
                              <option key={key} value={key}>{llmOptions[key].name}</option>
                          ))}
                      </select>
                      <select
                          className="flex-1 min-w-[120px] p-2 rounded-md border bg-garden-paper font-poetic text-sm"
                          value={currentUserChatModel}
                          onChange={e => setCurrentUserChatModel(e.target.value)}
                      >
                          {llmOptions[currentUserChatProvider]?.models.filter(model =>
                              modelHealth[model] !== 'offline' || model === adminDefaultModels.chat
                          ).map(model => {
                              const status = modelHealth[model];
                              const indicator = status === 'online' ? '🟢' : status === 'offline' ? '🔴' : '⚪';
                              return (
                                  <option key={model} value={model} disabled={status === 'offline'}>
                                      {indicator} {model} {status === 'offline' ? '(Offline — update admin default)' : ''}
                                  </option>
                              );
                          })}
                      </select>
                  </div>
                  <div className="flex flex-wrap items-center gap-4">
                      <div className="flex items-center gap-2">
                          <label className="font-poetic text-garden-inksoft text-sm whitespace-nowrap">Temp:</label>
                          <input
                              type="range"
                              min="0"
                              max="2"
                              step="0.1"
                              value={currentUserTemperature}
                              onChange={e => setCurrentUserTemperature(parseFloat(e.target.value))}
                              className="w-20"
                          />
                          <span className="text-xs text-garden-inksoft w-6">{currentUserTemperature}</span>
                      </div>
                      <div className="flex items-center gap-2">
                          <label className={`font-poetic text-sm whitespace-nowrap ${currentUserChatProvider === 'openai' ? 'text-garden-inksoft' : 'text-garden-inksoft'}`}>Top K:</label>
                          <input
                              type="range"
                              min="1"
                              max="100"
                              step="1"
                              value={currentUserTopK}
                              onChange={e => setCurrentUserTopK(parseInt(e.target.value))}
                              className="w-20"
                              disabled={currentUserChatProvider === 'openai'}
                          />
                          <span className={`text-xs w-6 ${currentUserChatProvider === 'openai' ? 'text-garden-inksoft' : 'text-garden-inksoft'}`}>{currentUserChatProvider === 'openai' ? '—' : currentUserTopK}</span>
                      </div>
                      <div className="flex items-center gap-2">
                          <label className="font-poetic text-garden-inksoft text-sm whitespace-nowrap">Top P:</label>
                          <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.05"
                              value={currentUserTopP}
                              onChange={e => setCurrentUserTopP(parseFloat(e.target.value))}
                              className="w-20"
                          />
                          <span className="text-xs text-garden-inksoft w-8">{currentUserTopP}</span>
                      </div>
                  </div>
              </div>

              {/* ── Voice ── */}
              <p className="font-poetic text-garden-inksoft text-xs font-semibold uppercase tracking-wider pt-2">Voice</p>

              {/* Voice Chat provider and model — response generation for voice sessions */}
              <div className="p-3 rounded-lg border bg-garden-paper2">
                  <label className="block font-poetic text-garden-ink font-semibold text-sm mb-2">Voice Chat</label>
                  <p className="font-poetic text-garden-inksoft text-xs mb-2">
                    Response model for voice sessions. Voice is latency-critical (speech can't start until the reply finishes) — prefer fast models with low variance.
                  </p>
                  <div className="flex flex-wrap items-center gap-2">
                      <select
                          className="flex-1 min-w-[120px] p-2 rounded-md border bg-garden-paper font-poetic text-sm"
                          value={currentUserVoiceChatProvider}
                          onChange={e => {
                              setCurrentUserVoiceChatProvider(e.target.value);
                              setCurrentUserVoiceChatModel(llmOptions[e.target.value].models[0]);
                          }}
                      >
                          {Object.keys(llmOptions).map(key => (
                              <option key={key} value={key}>{llmOptions[key].name}</option>
                          ))}
                      </select>
                      <select
                          className="flex-1 min-w-[120px] p-2 rounded-md border bg-garden-paper font-poetic text-sm"
                          value={currentUserVoiceChatModel}
                          onChange={e => setCurrentUserVoiceChatModel(e.target.value)}
                      >
                          {llmOptions[currentUserVoiceChatProvider]?.models.filter(model =>
                              modelHealth[model] !== 'offline' || model === adminDefaultModels.voiceChat
                          ).map(model => {
                              const status = modelHealth[model];
                              const indicator = status === 'online' ? '🟢' : status === 'offline' ? '🔴' : '⚪';
                              return (
                                  <option key={model} value={model} disabled={status === 'offline'}>
                                      {indicator} {model} {status === 'offline' ? '(Offline — update admin default)' : ''}
                                  </option>
                              );
                          })}
                      </select>
                  </div>
              </div>

              {/* Voice engine routing (Fase 5) */}
              <div className="p-3 rounded-lg border bg-garden-paper2">
                  <label className="block font-poetic text-garden-ink font-semibold text-sm mb-2">Voice engine</label>
                  <div className="flex items-center gap-3">
                      <Switch
                          checked={vpsVoiceOn}
                          disabled={!selectedAvatarId}
                          onCheckedChange={handleToggleVoiceBackend}
                          className="data-[state=checked]:bg-garden-moss"
                      />
                      {vpsVoiceOn ? (
                        <span className="font-poetic text-garden-inksoft text-xs">
                          VPS-hosted orchestrator (runs on this server; experimental)
                        </span>
                      ) : (
                        <span className="font-poetic text-garden-inksoft text-xs">
                          Agora ConvoAI (cloud, default)
                        </span>
                      )}
                  </div>
              </div>

              {/* Voice (Cartesia TTS) — per-avatar voice + synthesis language */}
              <div className="p-3 rounded-lg border bg-garden-paper2">
                  <label className="block font-poetic text-garden-ink font-semibold text-sm mb-2">Voice (Cartesia)</label>
                  <p className="font-poetic text-garden-inksoft text-xs mb-2">
                    Voices are accent-native — pick one matching the avatar's primary language. Applies to new voice sessions.
                  </p>
                  <div className="flex flex-wrap items-center gap-2">
                      <select
                          className="flex-1 min-w-[160px] p-2 rounded-md border bg-garden-paper font-poetic text-sm"
                          value={ttsVoiceSel}
                          onChange={e => setTtsVoiceSel(e.target.value)}
                      >
                          <option value="">Default (Miles — en)</option>
                          {ttsVoices.map(v => (
                              <option key={v.id} value={v.id}>{v.name} ({v.language})</option>
                          ))}
                      </select>
                      <input
                          className="w-24 p-2 rounded-md border bg-garden-paper font-poetic text-sm"
                          placeholder="lang (pt)"
                          value={ttsLangSel}
                          onChange={e => setTtsLangSel(e.target.value)}
                          title="Synthesis language hint (ISO code: en, pt, de …)"
                      />
                      <Button
                          className="font-poetic"
                          onClick={handleSaveTtsConfig}
                          disabled={ttsSaving || !selectedAvatarId}
                      >
                          {ttsSaving ? "Saving..." : "Save voice"}
                      </Button>
                  </div>
                  {ttsMsg && <p className="mt-1 text-xs text-garden-inksoft">{ttsMsg}</p>}
              </div>

              {/* ── Shared (both labs) ── */}
              <p className="font-poetic text-garden-inksoft text-xs font-semibold uppercase tracking-wider pt-2">Shared (text + voice)</p>

              {/* Text Query provider and model */}
              <div className="p-3 rounded-lg border bg-garden-paper2">
                  <label className="block font-poetic text-garden-ink font-semibold text-sm mb-2">Text Query</label>
                  <p className="font-poetic text-garden-inksoft text-xs mb-2">
                    Smaller models are faster — this model only generates retrieval keywords, so a small model (e.g. 8B) keeps responses quick.
                  </p>
                  <div className="flex flex-wrap items-center gap-2">
                      <select
                          className="flex-1 min-w-[120px] p-2 rounded-md border bg-garden-paper font-poetic text-sm"
                          value={currentUserTextQueryProvider}
                          onChange={e => {
                              setCurrentUserTextQueryProvider(e.target.value);
                              setCurrentUserTextQueryModel(llmOptions[e.target.value].models[0]);
                          }}
                      >
                          {Object.keys(llmOptions).map(key => (
                              <option key={key} value={key}>{llmOptions[key].name}</option>
                          ))}
                      </select>
                      <select
                          className="flex-1 min-w-[120px] p-2 rounded-md border bg-garden-paper font-poetic text-sm"
                          value={currentUserTextQueryModel}
                          onChange={e => setCurrentUserTextQueryModel(e.target.value)}
                      >
                          {llmOptions[currentUserTextQueryProvider]?.models.filter(model =>
                              modelHealth[model] !== 'offline' || model === adminDefaultModels.textQuery
                          ).map(model => {
                              const status = modelHealth[model];
                              const indicator = status === 'online' ? '🟢' : status === 'offline' ? '🔴' : '⚪';
                              return (
                                  <option key={model} value={model} disabled={status === 'offline'}>
                                      {indicator} {model} {status === 'offline' ? '(Offline — update admin default)' : ''}
                                  </option>
                              );
                          })}
                      </select>
                  </div>
              </div>

              {/* Sensor provider and model */}
              <div className="p-3 rounded-lg border bg-garden-paper2">
                  <label className="block font-poetic text-garden-ink font-semibold text-sm mb-2">Sensor</label>
                  <div className="flex flex-wrap items-center gap-2">
                      <select
                          className="flex-1 min-w-[120px] p-2 rounded-md border bg-garden-paper font-poetic text-sm"
                          value={currentUserSensorProvider}
                          onChange={e => {
                              setCurrentUserSensorProvider(e.target.value);
                              setCurrentUserSensorModel(llmOptions[e.target.value].models[0]);
                          }}
                      >
                          {Object.keys(llmOptions).map(key => (
                              <option key={key} value={key}>{llmOptions[key].name}</option>
                          ))}
                      </select>
                      <select
                          className="flex-1 min-w-[120px] p-2 rounded-md border bg-garden-paper font-poetic text-sm"
                          value={currentUserSensorModel}
                          onChange={e => setCurrentUserSensorModel(e.target.value)}
                      >
                          {llmOptions[currentUserSensorProvider]?.models.filter(model =>
                              modelHealth[model] !== 'offline' || model === adminDefaultModels.sensor
                          ).map(model => {
                              const status = modelHealth[model];
                              const indicator = status === 'online' ? '🟢' : status === 'offline' ? '🔴' : '⚪';
                              return (
                                  <option key={model} value={model} disabled={status === 'offline'}>
                                      {indicator} {model} {status === 'offline' ? '(Offline — update admin default)' : ''}
                                  </option>
                              );
                          })}
                      </select>
                  </div>
              </div>

              {/* Knowledge (RAG) memory pinning */}
              <div className="p-3 rounded-lg border bg-garden-paper2">
                  <label className="block font-poetic text-garden-ink font-semibold text-sm mb-2">Knowledge (RAG)</label>
                  <div className="flex items-center gap-3">
                      <Switch
                          checked={!!selectedAvatar?.ragPinned}
                          disabled={ragPinSaving || !selectedAvatarId}
                          onCheckedChange={handleToggleRagPinned}
                          className="data-[state=checked]:bg-garden-moss"
                      />
                      <span className="font-poetic text-garden-inksoft text-xs">
                        Keep knowledge in memory — removes the first-request delay after idle
                        periods. For high-traffic avatars; uses server RAM.
                      </span>
                  </div>
                  <div className="flex items-center gap-3 mt-3 pt-3 border-t border-garden-line/60">
                      <Switch
                          checked={keywordLlmOn}
                          disabled={!selectedAvatarId}
                          onCheckedChange={handleToggleKeywordMode}
                          className="data-[state=checked]:bg-garden-moss"
                      />
                      {keywordLlmOn ? (
                        <span className="font-poetic text-garden-inksoft text-xs">
                          Cloud keyword extraction (GWDG LLM) — context-aware, multilingual,
                          ~1s per message
                        </span>
                      ) : (
                        <span className="font-poetic text-garden-inksoft text-xs">
                          VPS keyword extraction (spaCy on this server) — fast, no network;
                          best for monolingual corpus (e.g. pt-only)
                        </span>
                      )}
                  </div>
              </div>

              {/* Admin defaults toggle */}
              <div className="flex items-center justify-center space-x-2 mt-2">
                  <label className="font-poetic text-garden-inksoft text-sm">Admin Defaults:</label>
                  <Switch
                      checked={hasLlmDefaults}
                      onCheckedChange={(checked) => {
                          if (checked) {
                              handleSaveLlmDefaults();
                          } else {
                              handleClearLlmDefaults();
                          }
                      }}
                      className="data-[state=checked]:bg-garden-moss"
                  />
                  <span className="text-xs text-garden-inksoft">
                      {hasLlmDefaults ? 'Saved as Admin default' : 'Not saved'}
                  </span>
              </div>

              {/* Model list freshness */}
              <p className="text-xs text-garden-inksoft text-center mt-1">
                {llmOptionsLastRefreshed
                  ? <>Model list last refreshed: <span className="text-garden-inksoft">{new Date(llmOptionsLastRefreshed).toLocaleString()}</span></>
                  : 'Model list: not yet refreshed this session'}
              </p>

            </div>
          )}
        </div>

        {/* Horizontal separator */}
        <hr className="w-full border-garden-line border-dashed" />

        {/* Collapsible API Keys Section */}
        <div className="w-full">
          <button
            className="flex items-center gap-2 font-poetic text-garden-inksoft cursor-pointer hover:text-garden-ink"
            onClick={() => setIntegrationsExpanded(!integrationsExpanded)}
          >
            <span className="text-lg">{integrationsExpanded ? '▼' : '▶'}</span>
            <span className="font-semibold">API Keys</span>
          </button>

          {integrationsExpanded && (
            <div className="mt-2 space-y-3">
              {/* LLM Providers group */}
              {llmProvidersList.filter(p => !p.hidden).length > 0 && (
                <div className="p-3 rounded-lg border bg-garden-paper2 space-y-2">
                  <label className="block font-poetic text-garden-ink font-semibold text-sm">LLM Providers</label>
                  {llmProvidersList.filter(p => !p.hidden).map(p => (
                    <div key={p.id} className="flex flex-wrap items-center gap-2">
                      <span className="font-poetic text-garden-inksoft text-xs w-40 shrink-0">{p.name}</span>
                      <span className="text-xs">
                        {p.key_set
                          ? <span className="text-garden-moss">set (••••{p.key_last4})</span>
                          : <span className="text-garden-amber">no key</span>}
                      </span>
                      <input
                        type="password"
                        className="flex-1 min-w-[160px] p-1 rounded border bg-garden-paper font-data text-xs"
                        placeholder="Enter new key to update"
                        value={providerKeyEdits[p.id] || ""}
                        onChange={e => setProviderKeyEdits(prev => ({ ...prev, [p.id]: e.target.value }))}
                      />
                    </div>
                  ))}
                </div>
              )}

              {/* Web Search group */}
              <div className="p-3 rounded-lg border bg-garden-paper2 space-y-2">
                <label className="block font-poetic text-garden-ink font-semibold text-sm">Web Search</label>
                {[
                  { key: "BRAVE_SEARCH_API_KEY", label: "Brave Search API key" },
                ].map(({ key, label }) => {
                  const info = integrations[key] || {};
                  return (
                    <div key={key} className="flex flex-wrap items-center gap-2">
                      <span className="font-poetic text-garden-inksoft text-xs w-40 shrink-0">{label}</span>
                      <span className="text-xs">
                        {info.set
                          ? <span className="text-garden-moss">set (••••{info.last4})</span>
                          : <span className="text-garden-clay">not set</span>}
                      </span>
                      <input
                        type="password"
                        className="flex-1 min-w-[160px] p-1 rounded border bg-garden-paper font-data text-xs"
                        placeholder="Enter new value to update"
                        value={integrationEdits[key] || ""}
                        onChange={e => setIntegrationEdits(prev => ({ ...prev, [key]: e.target.value }))}
                      />
                    </div>
                  );
                })}
              </div>

              {/* Voice Lab group */}
              <div className="p-3 rounded-lg border bg-garden-paper2 space-y-2">
                <label className="block font-poetic text-garden-ink font-semibold text-sm">Voice Lab</label>
                {[
                  { key: "AGORA_APP_ID", label: "Agora App ID" },
                  { key: "AGORA_APP_CERTIFICATE", label: "Agora App Certificate" },
                  { key: "AGORA_CUSTOMER_ID", label: "Agora Customer ID" },
                  { key: "AGORA_CUSTOMER_SECRET", label: "Agora Customer Secret" },
                  { key: "DEEPGRAM_API_KEY", label: "Deepgram API key" },
                  { key: "CARTESIA_API_KEY", label: "Cartesia API key" },
                ].map(({ key, label }) => {
                  const info = integrations[key] || {};
                  return (
                    <div key={key} className="flex flex-wrap items-center gap-2">
                      <span className="font-poetic text-garden-inksoft text-xs w-40 shrink-0">{label}</span>
                      <span className="text-xs">
                        {info.set
                          ? <span className="text-garden-moss">set (••••{info.last4})</span>
                          : <span className="text-garden-clay">not set</span>}
                      </span>
                      <input
                        type="password"
                        className="flex-1 min-w-[160px] p-1 rounded border bg-garden-paper font-data text-xs"
                        placeholder="Enter new value to update"
                        value={integrationEdits[key] || ""}
                        onChange={e => setIntegrationEdits(prev => ({ ...prev, [key]: e.target.value }))}
                      />
                    </div>
                  );
                })}
              </div>

              {/* Save integration keys */}
              <div className="flex items-center gap-3">
                <Button
                  className="font-poetic"
                  onClick={handleSaveIntegrations}
                  disabled={integrationSaving || (Object.keys(integrationEdits).length === 0 && Object.values(providerKeyEdits).every(v => !v))}
                >
                  {integrationSaving ? "Saving..." : "Save all changes"}
                </Button>
                {integrationMsg && (
                  <span className="text-xs text-garden-inksoft">{integrationMsg}</span>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Horizontal separator */}
        <hr className="w-full border-garden-line border-dashed" />

        {/* Collapsible Latency Analysis */}
        <div className="w-full">
          <button
            className="flex items-center gap-2 font-poetic text-garden-inksoft cursor-pointer hover:text-garden-ink"
            onClick={() => setLatencyExpanded(!latencyExpanded)}
          >
            <span className="text-lg">{latencyExpanded ? '▼' : '▶'}</span>
            <span className="font-semibold">Latency Analysis</span>
            {lastLatency && (
              <span className="text-xs text-garden-inksoft font-data">
                {(lastLatency.roundTripMs / 1000).toFixed(2)}s
              </span>
            )}
          </button>

          {latencyExpanded && (
            <div className="mt-2 p-3 rounded-lg border bg-garden-paper2">
              {!lastLatency ? (
                <p className="font-poetic text-garden-inksoft text-sm">
                  Send a message to see its latency breakdown. its latency breakdown.
                </p>
              ) : (() => {
                const t = lastLatency.timings;
                const segments = [
                  { label: "Loading RAG index into RAM", ms: t.index_load_ms || 0, color: "#9333EA" },
                  { label: "Keyword generation (LLM)", ms: t.keyword_gen_ms || 0, color: "#CA8A04" },
                  { label: "Knowledge retrieval (RAG)", ms: t.rag_retrieval_ms || 0, color: "#0E7490" },
                  { label: "Web search", ms: t.web_search_ms || 0, color: "#DB2777" },
                  { label: "Sensor snapshot", ms: t.sensor_snapshot_ms || 0, color: "#15803D" },
                  { label: "Avatar response (LLM)", ms: t.main_llm_ms || 0, color: "#1D4ED8" },
                  { label: "Sensor analysis tool", ms: t.sensor_tool_ms || 0, color: "#EA580C" },
                ];
                const attributed = segments.reduce((acc, s) => acc + s.ms, 0);
                const backendOther = Math.max((t.total_backend_ms || 0) - attributed, 0);
                if (backendOther > 0) segments.push({ label: "Backend overhead", ms: backendOther, color: "#A8A29E" });
                const network = Math.max(lastLatency.roundTripMs - (t.total_backend_ms || 0), 0);
                if (network > 0) segments.push({ label: "Network + transport", ms: network, color: "#57534E" });
                return <LatencyBreakdown segments={segments} totalMs={lastLatency.roundTripMs} />;
              })()}
            </div>
          )}
        </div>

        {/* Horizontal separator */}
        <hr className="w-full border-garden-line border-dashed" />

        {/* Collapsible Backend Log */}
        <div className="w-full">
          <button
            className="flex items-center gap-2 font-poetic text-garden-inksoft cursor-pointer hover:text-garden-ink"
            onClick={() => setLogExpanded(!logExpanded)}
          >
            <span className="text-lg">{logExpanded ? '▼' : '▶'}</span>
            <span className="font-semibold">Backend Log</span>
          </button>

          {logExpanded && (
            <div className="mt-2">
              <Card className="bg-garden-ink border-garden-ink overflow-hidden">
                <pre
                  ref={backendLogPreRef}
                  className="overflow-y-auto px-3 md:px-4 py-3 text-[#EDE7DA] text-xs leading-relaxed"
                  style={{ fontFamily: "'IBM Plex Mono', monospace", maxHeight: "50vh", whiteSpace: "pre-wrap", wordBreak: "break-all" }}
                >
                  {backendLog || "Loading..."}
                  <div ref={backendLogRef} />
                </pre>
              </Card>
            </div>
          )}
        </div>

        {/* Horizontal separator */}
        <hr className="w-full border-garden-line border-dashed" />

        {selectedAvatar && (
            <div className="flex gap-2 items-center justify-center mt-2">
              <Button
                variant="outline"
                className="font-poetic"
                onClick={() => handleRefresh('prompt')}
                disabled={!selectedAvatarId || refreshState.prompt !== 'idle'}
              >
                {refreshState.prompt === 'loading' ? 'Refreshing...' : refreshState.prompt === 'success' ? 'Refreshed!' : refreshState.prompt === 'error' ? 'Error!' : 'Refresh Prompt'}
              </Button>
              <Button
                variant="outline"
                className="font-poetic"
                onClick={() => handleRefresh('embeddings')}
                disabled={!selectedAvatarId || refreshState.embeddings !== 'idle'}
              >
                {refreshState.embeddings === 'loading' ? 'Refreshing...' : refreshState.embeddings === 'success' ? 'Refreshed!' : refreshState.embeddings === 'error' ? 'Error!' : 'Refresh Avatar Context Files'}
              </Button>
            </div>
          )}

          {selectedAvatar && (
            <div className="mt-4 text-xs md:text-sm text-garden-inksoft space-y-1 break-all">
              <div><span className="font-semibold">Prompt:</span> {selectedAvatar.systemPromptUrl || "—"}</div>
              <div><span className="font-semibold">Context:</span> {selectedAvatar.contextDocsUrl || "—"}</div>
              <div><span className="font-semibold">Sensors:</span> {selectedAvatar.sensorApiUrl || "—"}</div>
              <div><span className="font-semibold">Languages in context documents:</span> {Array.isArray(selectedAvatar.ragLanguages) ? selectedAvatar.ragLanguages.join(", ") : (selectedAvatar.ragLanguages || "—")}</div>
              <div><span className="font-semibold">API:</span> https://avatars.sympoiesis.xyz/api/chat?avatar={selectedAvatar.id}</div>
            </div>
          )}
          {!selectedAvatar && (
            <div className="mt-2 text-xs text-garden-clay">
              Please select or create an avatar to start chatting.
            </div>
          )}
      </div>

      {isDebateMode && (
        <div className="w-full max-w-5xl mb-4 px-4">
          <label className="block mb-1 font-poetic text-garden-ink">Choose a topic:</label>
          <select
            className="w-full p-2 rounded-md border bg-garden-paper font-poetic"
            value={selectedTopic}
            onChange={e => { setSelectedTopic(e.target.value); }}
          >
            <option value="">-- select --</option>
            {topics.map((t, i) => (
              <option key={i} value={t}>{t}</option>
            ))}
          </select>
          {selectedTopic && (
            <div className="mt-2 p-2 md:p-3 bg-garden-paper rounded-md border text-sm md:text-base text-garden-inksoft font-poetic">
              {topicDescriptions[selectedTopic]}
            </div>
          )}
        </div>
      )}

      <motion.div className="px-2 md:px-4 w-full max-w-5xl mx-auto flex-1 overflow-visible">
        <div className={`flex flex-col md:flex-row ${isDebateMode ? 'md:space-x-4' : ''} min-h-0`}>
          <div className={`${isDebateMode ? 'md:flex-1' : 'w-full'} min-h-0 rounded-2xl mb-4 md:mb-0`}>
            <Card className="flex flex-col h-full min-h-0 shadow-lg bg-card">
              {/* scrollable messages */}
              <div
                onWheel={e => e.stopPropagation()}
                className="flex-1 h-[60vh] md:h-[70vh] min-h-0 overflow-y-auto px-3 md:px-8 py-4 md:py-6"
              >
                {messages.map((msg, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className={`flex ${msg.sender === 'avatar' ? 'justify-start' : 'justify-end'} mb-4`}
                  >
                    <div
                      className={`max-w-xs md:max-w-lg px-4 py-3 rounded-lg text-sm md:text-lg whitespace-pre-wrap leading-relaxed ${
                        msg.sender === 'avatar'
                          ? 'bg-garden-mosstint text-garden-ink border border-garden-line'
                          : 'bg-card text-garden-ink border border-garden-line'
                      }`}
                    >
                      {msg.text}
                    </div>
                  </motion.div>
                ))}

                {isThinking && (
                  <motion.div
                    className="font-poetic text-garden-moss self-start mb-4"
                    animate={{ opacity: [0.3, 1, 0.3], x: [0, 2, -2, 0] }}
                    transition={{ repeat: Infinity, duration: 2 }}
                  >
                    the avatar contemplates...
                  </motion.div>
                )}

                <div ref={chatEndRef} />
              </div>

              {/* offline model warning */}
              {offlineBlockers.length > 0 && (
                <div className="px-3 md:px-6 py-2 bg-garden-clay/10 border-t border-garden-clay/30 text-garden-clay text-xs font-poetic">
                  ⚠️ {offlineBlockers.map(b => `${b.role} model "${b.model}"`).join(' and ')} {offlineBlockers.length > 1 ? 'are' : 'is'} offline. Open LLM Config to select an active model.
                </div>
              )}

              {/* input bar */}
              <div className="flex items-center gap-2 px-3 md:px-6 py-3 md:py-4 border-t bg-garden-paper2">
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.6 }}
                  className="flex-1"
                >
                  <Input
                    className="w-full rounded-full font-poetic bg-garden-paper text-garden-ink"
                    style={{ color: '#2B2B24' }}
                    placeholder={!selectedAvatar ? "Select or create an avatar to begin..." : "Speak with the avatar..."}
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => !chatDisabled && e.key === 'Enter' && handleSubmit()}
                    disabled={!selectedAvatar}
                  />
                </motion.div>
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.6 }}
                >
                  <Button
                    onClick={handleSubmit}
                    className="rounded-md px-6 py-2 font-poetic bg-garden-moss text-garden-paper hover:bg-garden-mossdeep disabled:opacity-50"
                    disabled={chatDisabled}
                  >
                    Flow
                  </Button>
                </motion.div>
              </div>
            </Card>
          </div>

          {isDebateMode && (
            <div className="w-full md:w-1/3 bg-garden-paper rounded-xl border border-garden-line shadow-sm p-3 md:p-4 h-[40vh] md:h-[60vh] overflow-y-auto">
              <h4 className="font-poetic text-base md:text-lg font-bold mb-2">Debate Summary</h4>
              <div
                className="text-xs md:text-sm text-garden-inksoft whitespace-pre-wrap"
                dangerouslySetInnerHTML={{ __html: debateSummary }}
              />
            </div>
          )}
        </div>
      </motion.div>

      {/* Simple avatar create/edit panel (inline "modal") */}
      {avatarFormOpen && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-40 p-2">
          <div className="bg-garden-paper rounded-xl border border-garden-line shadow-xl w-full max-w-lg max-h-[90vh] overflow-y-auto p-4 md:p-6">
            <h2 className="text-lg md:text-xl font-poetic mb-4">
              {avatarFormMode === "create" ? "Create New Avatar" : "Edit Avatar"}
            </h2>
            <form onSubmit={handleAvatarFormSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-semibold mb-1">Avatar name</label>
                <Input
                  className="border border-garden-line rounded-md px-3 py-2 placeholder:text-garden-inksoft/60"
                  style={{
                    backgroundColor: '#EDE7DA',
                    color: "#2B2B24",
                  }}
                  value={avatarForm.name}
                  onChange={e => handleAvatarFormChange("name", e.target.value)}
                  placeholder="e.g. Lahn River, Forest Spirit..."
                />
              </div>
              <div>
                <label className="block text-sm font-semibold mb-1">
                  Link to system prompt
                </label>
                <Input
                  className="border border-garden-line rounded-md px-3 py-2 placeholder:text-garden-inksoft/60"
                  style={{
                    backgroundColor: '#EDE7DA',
                    color: "#2B2B24",
                  }}
                  value={avatarForm.systemPromptUrl}
                  onChange={e => handleAvatarFormChange("systemPromptUrl", e.target.value)}
                  placeholder="Google Docs Link (e.g., https://docs.google.com/...)"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold mb-1">
                  Link to context documents
                </label>
                <Input
                  className="border border-garden-line rounded-md px-3 py-2 placeholder:text-garden-inksoft/60"
                  style={{
                    backgroundColor: '#EDE7DA',
                    color: "#2B2B24",
                  }}
                  value={avatarForm.contextDocsUrl}
                  onChange={e => handleAvatarFormChange("contextDocsUrl", e.target.value)}
                  placeholder="Google Drive Folder Link (e.g., https://drive.google.com/drive/folders/...)"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold mb-1">
                  Link to sensor data API
                </label>
                <Input
                  className="border border-garden-line rounded-md px-3 py-2 placeholder:text-garden-inksoft/60"
                  style={{
                    backgroundColor: '#EDE7DA',
                    color: "#2B2B24",
                  }}
                  value={avatarForm.sensorApiUrl}
                  onChange={e => handleAvatarFormChange("sensorApiUrl", e.target.value)}
                  placeholder="https://…/sensors"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold mb-1">
                  Sensor data description
                </label>
                <Input
                  className="border border-garden-line rounded-md px-3 py-2 placeholder:text-garden-inksoft/60"
                  style={{
                    backgroundColor: '#EDE7DA',
                    color: "#2B2B24",
                  }}
                  value={avatarForm.sensorDescription}
                  onChange={e => handleAvatarFormChange("sensorDescription", e.target.value)}
                  placeholder="e.g. Provides live pH, temperature, dissolved oxygen, and conductivity data"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold mb-1">
                  What languages are your context documents in?
                </label>
                <Input
                  className="border border-garden-line rounded-md px-3 py-2 placeholder:text-garden-inksoft/60"
                  style={{
                    backgroundColor: '#EDE7DA',
                    color: "#2B2B24",
                  }}
                  value={avatarForm.ragLanguages}
                  onChange={e => handleAvatarFormChange("ragLanguages", e.target.value)}
                  placeholder="en, de, fr, pt"
                />
                <p className="text-xs text-garden-inksoft mt-1">
                  Enter language codes separated by commas. This helps optimize RAG queries for your document languages.
                </p>
              </div>

              {avatarError && (
                <div className="text-sm text-garden-clay">{avatarError}</div>
              )}

              <div className="flex justify-end gap-2 mt-4">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setAvatarFormOpen(false)}
                  disabled={avatarSaving}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={avatarSaving}>
                  {avatarSaving
                    ? avatarFormMode === "create"
                      ? "Saving... (Might take a few minutes to prepare your context data)"
                      : "Saving..."
                    : avatarFormMode === "create"
                      ? "Create"
                      : "Save changes"}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* LLM Provider create/edit panel */}
      {providerFormOpen && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-40 p-2">
          <div className="bg-garden-paper rounded-xl border border-garden-line shadow-xl w-full max-w-lg max-h-[90vh] overflow-y-auto p-4 md:p-6">
            <h2 className="text-lg md:text-xl font-poetic mb-4">
              Add New LLM Provider
            </h2>
            <form onSubmit={handleProviderFormSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-semibold mb-1">Provider name</label>
                <Input
                  className="border border-garden-line rounded-md px-3 py-2 placeholder:text-garden-inksoft/60"
                  style={{
                    backgroundColor: '#EDE7DA',
                    color: "#2B2B24",
                  }}
                  value={providerForm.name}
                  onChange={e => handleProviderFormChange("name", e.target.value)}
                  placeholder="e.g. Anthropic, Azure OpenAI, HuggingFace..."
                />
              </div>
              <div>
                <label className="block text-sm font-semibold mb-1">
                  API base URL
                </label>
                <Input
                  className="border border-garden-line rounded-md px-3 py-2 placeholder:text-garden-inksoft/60"
                  style={{
                    backgroundColor: '#EDE7DA',
                    color: "#2B2B24",
                  }}
                  value={providerForm.api_base}
                  onChange={e => handleProviderFormChange("api_base", e.target.value)}
                  placeholder="https://api.example.com/v1"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold mb-1">
                  API key
                </label>
                <Input
                  type="password"
                  className="border border-garden-line rounded-md px-3 py-2 placeholder:text-garden-inksoft/60"
                  style={{
                    backgroundColor: '#EDE7DA',
                    color: "#2B2B24",
                  }}
                  value={providerForm.api_key}
                  onChange={e => handleProviderFormChange("api_key", e.target.value)}
                  placeholder="sk-..."
                />
              </div>
              <div>
                <label className="block text-sm font-semibold mb-1">
                  Models (comma-separated)
                </label>
                <Input
                  className="border border-garden-line rounded-md px-3 py-2 placeholder:text-garden-inksoft/60"
                  style={{
                    backgroundColor: '#EDE7DA',
                    color: "#2B2B24",
                  }}
                  value={providerForm.models}
                  onChange={e => handleProviderFormChange("models", e.target.value)}
                  placeholder="claude-3-opus, claude-3-sonnet, claude-3-haiku"
                />
              </div>

              {providerError && (
                <div className="text-sm text-garden-clay">{providerError}</div>
              )}

              <div className="flex justify-end gap-2 mt-4">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setProviderFormOpen(false)}
                  disabled={providerSaving}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={providerSaving}>
                  {providerSaving ? "Adding..." : "Add Provider"}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
