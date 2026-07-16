const API_BASES = ["http://127.0.0.1:8000", "http://localhost:8000"];
const DEFAULT_API_BASE = "http://127.0.0.1:8000";

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
  chrome.scripting.executeScript({
    target: { tabId },
    files: ["content.js"],
  });
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

  if (msg && msg.type === "OPEN_UPLOAD_PAGE") {
    const url = chrome.runtime.getURL("upload.html");
    chrome.tabs.create({ url });
    sendResponse({ ok: true });
    return true;
  }

  if (msg && msg.type === "ANALYZE_CURRENT_TAB") {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs && tabs[0];
      if (!tab || !tab.id) {
        sendResponse({ ok: false, error: "No active tab" });
        return;
      }
      recordLastTab(tab.id);
      injectContent(tab.id);
      sendResponse({ ok: true });
    });
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
    const headers = { "Content-Type": "application/json", Authorization: `Bearer ${token}` };
    fetch(`${DEFAULT_API_BASE}/analyze_and_save`, {
      method: "POST",
      mode: "cors",
      headers,
      body: JSON.stringify({
        url: msg.url || "",
        title: msg.title || "",
        page_text: msg.page_text || "",
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
        chrome.storage.local.set({ last_result: data });
        chrome.runtime.sendMessage({ type: "ANALYSIS_FINISHED", ok: true, data });
        sendResponse({ ok: true, data });
      })
      .catch((err) => {
        console.error("JD Extractor error:", err);
        chrome.runtime.sendMessage({ type: "ANALYSIS_FINISHED", ok: false, error: String(err) });
        sendResponse({ ok: false, error: String(err) });
      });
  });
  return true;
});
