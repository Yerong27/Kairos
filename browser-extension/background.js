const API_BASES = ["http://127.0.0.1:8000", "http://localhost:8000"];
const DEFAULT_API_BASE = "http://127.0.0.1:8000";
const ANALYSIS_STATE_KEY = "analysis_state";
const ACTIVE_STATE_MAX_AGE_MS = 4 * 60 * 1000;
const EXTRACTION_STATE_MAX_AGE_MS = 20 * 1000;
const activeRequests = new Set();

function enableSidePanel() {
  if (!chrome.sidePanel || !chrome.sidePanel.setPanelBehavior) return;
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch((error) => {
    console.error("Kairos side panel setup failed:", error);
  });
}

chrome.runtime.onInstalled.addListener(() => {
  enableSidePanel();
  chrome.storage.local.get([ANALYSIS_STATE_KEY], (result) => {
    const state = result && result[ANALYSIS_STATE_KEY];
    if (state && ["extracting", "analyzing"].includes(state.status)) {
      saveAnalysisState({
        ...state,
        status: "error",
        message: "Kairos was reloaded. The previous request stopped; you can retry now.",
      });
    }
  });
});
chrome.runtime.onStartup.addListener(enableSidePanel);
enableSidePanel();

function analysisJobKey(data) {
  const rawUrl = String((data && data.url) || "").trim();
  if (rawUrl) {
    try {
      const url = new URL(rawUrl);
      return `${url.origin}${url.pathname}`;
    } catch (_error) {
      return rawUrl.split("?")[0];
    }
  }
  return [data && data.title, data && data.company].filter(Boolean).join("|").toLowerCase();
}

function saveAnalysisState(state) {
  chrome.storage.local.set({
    [ANALYSIS_STATE_KEY]: { ...state, updated_at: Date.now() },
  });
}

function getAnalysisState(callback) {
  chrome.storage.local.get([ANALYSIS_STATE_KEY], (result) => {
    callback((result && result[ANALYSIS_STATE_KEY]) || null);
  });
}

function stateIsActive(state) {
  if (!state || !["extracting", "analyzing"].includes(state.status)) return false;
  const updatedAt = Number(state.updated_at || state.started_at || 0);
  const maxAge = state.status === "extracting"
    ? EXTRACTION_STATE_MAX_AGE_MS
    : ACTIVE_STATE_MAX_AGE_MS;
  return updatedAt > 0 && Date.now() - updatedAt < maxAge;
}

function getStoredToken(cb) {
  chrome.storage.local.get(["user_token"], (result) => {
    const token = result && result.user_token ? String(result.user_token) : "";
    cb(token);
  });
}

function setStoredToken(token, meta) {
  const payload = { user_token: token };
  if (meta && typeof meta === "object") {
    payload.db_id = meta.database_id || "";
    payload.db_name = meta.database_name || "";
  }
  chrome.storage.local.set(payload);
}

function recordLastTab(tabId) {
  if (chrome.storage.session && chrome.storage.session.set) {
    chrome.storage.session.set({ last_jd_tab_id: tabId });
  } else {
    chrome.storage.local.set({ last_jd_tab_id: tabId });
  }
}

function getLastTab(cb) {
  const getStore = chrome.storage.session && chrome.storage.session.get
    ? chrome.storage.session.get.bind(chrome.storage.session)
    : chrome.storage.local.get.bind(chrome.storage.local);
  getStore(["last_jd_tab_id"], (result) => {
    cb(result && result.last_jd_tab_id ? result.last_jd_tab_id : null);
  });
}

function injectContent(tabId) {
  return chrome.scripting.executeScript({
    target: { tabId },
    files: ["content.js"],
  });
}

function readableInjectionError(error) {
  const detail = String(error && error.message ? error.message : error || "");
  if (/cannot access|permission|host/i.test(detail)) {
    return "Kairos could not access this job tab. Reload the extension, then reopen the job page.";
  }
  return detail ? `Could not read the job page: ${detail}` : "Could not read the job page.";
}

function supportedJobSite(rawUrl) {
  try {
    const url = new URL(rawUrl);
    if (url.protocol !== "https:") return "";
    const host = url.hostname.toLowerCase();
    if ((host === "www.linkedin.com" || host === "linkedin.com") && /^\/jobs\//i.test(url.pathname)) {
      return "linkedin";
    }
    if ((host === "www.seek.com.au" || host === "seek.com.au") && /^\/job\/\d+\/?$/i.test(url.pathname)) {
      return "seek";
    }
  } catch (_error) {
    // An invalid or unsupported URL is not a job page.
  }
  return "";
}

function openNotionStart() {
  chrome.tabs.create({ url: `${DEFAULT_API_BASE}/notion/start` });
}

chrome.webNavigation.onCompleted.addListener((details) => {
  if (!details.url || !details.tabId) {
    return;
  }
  const url = new URL(details.url);
  if (!API_BASES.includes(url.origin) || url.pathname !== "/notion/done") {
    return;
  }
  const code = url.searchParams.get("code") || "";
  if (!code) {
    return;
  }
  fetch(`${url.origin}/auth/exchange`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code }),
  })
    .then(async (res) => {
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Authentication failed (${res.status})`);
      }
      return data;
    })
    .then((data) => {
      if (data && data.user_token) {
        setStoredToken(data.user_token, data);
        chrome.runtime.sendMessage({ type: "AUTH_UPDATED", data });
        chrome.tabs.remove(details.tabId);
        getLastTab((tabId) => {
          if (tabId) {
            injectContent(tabId);
          }
        });
      }
    })
    .catch((err) => {
      console.error("JD Extractor auth exchange error:", err);
    });
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg && msg.type === "OPEN_NOTION_CONNECT") {
    openNotionStart();
    sendResponse({ ok: true });
    return true;
  }

  if (msg && msg.type === "ANALYZE_CURRENT_TAB") {
    getAnalysisState((state) => {
      if (stateIsActive(state)) {
        sendResponse({ ok: true, in_progress: true, state });
        return;
      }
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const tab = tabs && tabs[0];
        if (!tab || !tab.id) {
          sendResponse({ ok: false, error: "No active tab" });
          return;
        }
        if (!supportedJobSite(String(tab.url || ""))) {
          const error = "Open a LinkedIn or SEEK job page before analyzing.";
          saveAnalysisState({
            status: "error",
            title: tab.title || "Current page",
            url: tab.url || "",
            message: error,
          });
          sendResponse({ ok: false, error });
          return;
        }
        recordLastTab(tab.id);
        const extractionState = {
          status: "extracting",
          title: tab.title || "Current page",
          url: tab.url || "",
          started_at: Date.now(),
          message: "Reading the job page…",
        };
        saveAnalysisState(extractionState);
        injectContent(tab.id)
          .then(() => sendResponse({ ok: true }))
          .catch((error) => {
            const message = readableInjectionError(error);
            saveAnalysisState({ ...extractionState, status: "error", message });
            sendResponse({ ok: false, error: message });
          });
      });
    });
    return true;
  }

  if (msg && msg.type === "JD_EXTRACTION_FAILED") {
    const error = String(msg.error || "Could not extract the job description.");
    saveAnalysisState({
      status: "error",
      title: msg.title || "Job page",
      url: msg.url || "",
      message: error,
    });
    sendResponse({ ok: false, error });
    return true;
  }

  if (!msg || msg.type !== "JD_EXTRACT") {
    return;
  }

  getStoredToken((token) => {
    if (!token) {
      openNotionStart();
      sendResponse({ ok: false, error: "Notion auth required" });
      return;
    }
    const jobKey = analysisJobKey(msg);
    getAnalysisState((existingState) => {
      if (
        activeRequests.has(jobKey)
        || (stateIsActive(existingState) && existingState.job_key === jobKey && existingState.status === "analyzing")
      ) {
        sendResponse({ ok: true, in_progress: true });
        return;
      }

      activeRequests.add(jobKey);
      const stateBase = {
        job_key: jobKey,
        title: msg.title || "Job page",
        company: msg.company || "",
        location: msg.location || "",
        url: msg.url || "",
        started_at: Date.now(),
      };
      saveAnalysisState({ ...stateBase, status: "analyzing", message: "Analyzing and saving to Notion…" });
      chrome.runtime.sendMessage({
        type: "EXTRACTION_READY",
        data: {
          ...(msg.extraction_meta || {}),
          title: msg.title || "",
          company: msg.company || "",
          location: msg.location || "",
        },
      });
      const headers = { "Content-Type": "application/json", Authorization: `Bearer ${token}` };
      fetch(`${DEFAULT_API_BASE}/analyze_and_save`, {
        method: "POST",
        mode: "cors",
        headers,
        body: JSON.stringify({
          url: msg.url || "",
          title: msg.title || "",
          company: msg.company || "",
          location: msg.location || "",
          page_text: msg.page_text || "",
          extraction_meta: msg.extraction_meta || {},
          use_v3: true,
          output_language: "en",
        }),
      })
        .then(async (res) => {
          const data = await res.json().catch(() => ({}));
          if (!res.ok) {
            const detail = data && data.detail ? data.detail : `Backend error (${res.status})`;
            throw new Error(detail);
          }
          return data;
        })
        .then((data) => {
          saveAnalysisState({
            ...stateBase,
            status: "success",
            message: "Analysis saved to Notion.",
            notion_url: data && data.notion_url ? data.notion_url : "",
            score: data && data.final_score,
            recommendation: data && data.should_apply,
          });
          chrome.storage.local.set({ last_result: data });
          chrome.runtime.sendMessage({ type: "ANALYSIS_FINISHED", ok: true, data });
          sendResponse({ ok: true, data });
        })
        .catch((err) => {
          const error = String(err && err.message ? err.message : err).replace(/^Error:\s*/, "");
          console.error("JD Extractor error:", err);
          saveAnalysisState({ ...stateBase, status: "error", message: error });
          chrome.runtime.sendMessage({ type: "ANALYSIS_FINISHED", ok: false, error });
          sendResponse({ ok: false, error });
        })
        .finally(() => {
          activeRequests.delete(jobKey);
        });
    });
  });
  return true;
});
