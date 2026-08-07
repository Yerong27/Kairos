const API_BASE = "http://127.0.0.1:8000";

const notionStatus = document.getElementById("notion-status");
const notionDot = document.getElementById("notion-dot");
const dbName = document.getElementById("db-name");
const resumeStatus = document.getElementById("resume-status");
const resumeDot = document.getElementById("resume-dot");
const resumeMeta = document.getElementById("resume-meta");
const uploadBtn = document.getElementById("upload-btn");
const uploadMsg = document.getElementById("upload-msg");
const resumeFile = document.getElementById("resume-file");
const analyzeBtn = document.getElementById("analyze-btn");
const analyzeMsg = document.getElementById("analyze-msg");
const notionMsg = document.getElementById("notion-msg");
const connectBtn = document.getElementById("connect-btn");
const ANALYSIS_STATE_KEY = "analysis_state";
const ACTIVE_STATE_MAX_AGE_MS = 4 * 60 * 1000;
const EXTRACTION_STATE_MAX_AGE_MS = 20 * 1000;
let profileIsReady = false;
let analysisIsRunning = false;
let analysisStateTimer = null;

function updateAnalyzeAvailability() {
  analyzeBtn.disabled = !profileIsReady || analysisIsRunning;
}

function renderAnalysisState(state) {
  if (!state || !state.status) return;
  if (analysisStateTimer) {
    clearTimeout(analysisStateTimer);
    analysisStateTimer = null;
  }
  const role = [state.title, state.company].filter(Boolean).join(" · ");
  const updatedAt = Number(state.updated_at || state.started_at || 0);
  const maxAge = state.status === "extracting"
    ? EXTRACTION_STATE_MAX_AGE_MS
    : ACTIVE_STATE_MAX_AGE_MS;
  const fresh = updatedAt > 0 && Date.now() - updatedAt < maxAge;
  analysisIsRunning = ["extracting", "analyzing"].includes(state.status) && fresh;
  updateAnalyzeAvailability();

  if (["extracting", "analyzing"].includes(state.status) && !fresh) {
    analyzeMsg.textContent = `${role ? `${role} — ` : ""}${state.status === "extracting" ? "Kairos could not read the page." : "The previous request stopped."} You can retry.`;
  } else if (state.status === "success") {
    const result = [state.recommendation, state.score != null ? `${state.score}/100` : ""].filter(Boolean).join(" · ");
    analyzeMsg.textContent = `${role ? `${role} — ` : ""}Saved to Notion${result ? ` (${result})` : ""}.`;
  } else if (state.status === "error") {
    analyzeMsg.textContent = `${role ? `${role} — ` : ""}${state.message || "Analysis failed."}`;
  } else {
    analyzeMsg.textContent = `${role ? `${role} — ` : ""}${state.message || "Analyzing…"}`;
  }

  if (analysisIsRunning) {
    const remaining = Math.max(50, maxAge - (Date.now() - updatedAt) + 50);
    analysisStateTimer = setTimeout(() => renderAnalysisState(state), remaining);
  }
}

function restoreAnalysisState() {
  chrome.storage.local.get([ANALYSIS_STATE_KEY], (result) => {
    renderAnalysisState(result && result[ANALYSIS_STATE_KEY]);
  });
}

function setTone(element, tone) {
  if (tone) element.dataset.tone = tone;
  else delete element.dataset.tone;
}

function getToken(cb) {
  chrome.storage.local.get(["user_token"], (result) => {
    const token = result && result.user_token ? String(result.user_token) : "";
    cb(token);
  });
}

function setStatusDisconnected() {
  profileIsReady = false;
  notionStatus.textContent = "Not connected";
  setTone(notionDot, "warning");
  dbName.textContent = "Connect to choose a database";
  resumeStatus.textContent = "Connect Notion first";
  setTone(resumeDot, "warning");
  resumeMeta.textContent = "PDF, DOCX, or TXT";
  notionMsg.textContent = "";
  uploadMsg.textContent = "";
  analyzeMsg.textContent = "";
  uploadBtn.disabled = true;
  updateAnalyzeAvailability();
  connectBtn.textContent = "Connect";
  connectBtn.disabled = false;
}

function refreshStatus() {
  notionMsg.textContent = "";
  getToken((token) => {
    if (!token) {
      setStatusDisconnected();
      return;
    }
    fetch(`${API_BASE}/status`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error("status_error");
        }
        return res.json();
      })
      .then((data) => {
        notionStatus.textContent = data.notion_connected ? "Connected" : "Not connected";
        setTone(notionDot, data.notion_connected ? "success" : "warning");
        dbName.textContent = data.database_name || "No database selected";
        connectBtn.textContent = data.notion_connected ? "Change" : "Connect";
        connectBtn.disabled = false;
        notionMsg.textContent = "";
        if (data.resume_present) {
          const profileReady = Boolean(data.candidate_profile_current);
          const localFallback = data.candidate_profile_status === "degraded";
          resumeStatus.textContent = localFallback
            ? "Resume ready (local profile)"
            : profileReady
              ? "Resume ready"
              : "Profile needs attention";
          setTone(resumeDot, localFallback ? "warning" : profileReady ? "success" : "error");
          const meta = [];
          if (data.resume_filename) meta.push(data.resume_filename);
          if (data.resume_uploaded_at) meta.push(new Date(data.resume_uploaded_at * 1000).toLocaleDateString());
          resumeMeta.textContent = meta.join(" • ");
          uploadBtn.disabled = false;
          profileIsReady = profileReady;
          updateAnalyzeAvailability();
          analyzeMsg.textContent = localFallback
            ? "AI enrichment timed out. Local resume evidence will be used; re-upload later to retry."
            : profileReady
              ? ""
              : "Re-upload the resume to create or retry its Candidate Profile.";
          restoreAnalysisState();
        } else {
          resumeStatus.textContent = "No resume uploaded";
          setTone(resumeDot, "warning");
          resumeMeta.textContent = "PDF, DOCX, or TXT";
          uploadBtn.disabled = false;
          profileIsReady = false;
          updateAnalyzeAvailability();
          analyzeMsg.textContent = "";
          restoreAnalysisState();
        }
      })
      .catch(() => {
        notionStatus.textContent = "Backend offline";
        setTone(notionDot, "error");
        setTone(resumeDot, "error");
        dbName.textContent = "Start Kairos at 127.0.0.1:8000";
        resumeStatus.textContent = "Unavailable";
        resumeMeta.textContent = "";
        notionMsg.textContent = "";
        uploadMsg.textContent = "";
        analyzeMsg.textContent = "";
        uploadBtn.disabled = true;
        profileIsReady = false;
        updateAnalyzeAvailability();
      });
  });
}

connectBtn.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "OPEN_NOTION_CONNECT" });
});

function uploadSelectedResume() {
  uploadMsg.textContent = "";
  const file = resumeFile.files && resumeFile.files[0];
  if (!file) {
    return;
  }
  uploadBtn.disabled = true;
  uploadBtn.textContent = "Creating profile…";
  resumeFile.disabled = true;
  resumeMeta.textContent = file.name;

  getToken((token) => {
    if (!token) {
      uploadMsg.textContent = "Connect Notion first.";
      uploadBtn.disabled = false;
      uploadBtn.textContent = "Choose file";
      resumeFile.disabled = false;
      return;
    }
    const form = new FormData();
    form.append("file", file);
    fetch(`${API_BASE}/resume/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: form,
      keepalive: true,
    })
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `Upload failed (${res.status})`);
        return data;
      })
      .then((data) => {
        if (!data || data.status !== "saved") {
          uploadMsg.textContent = "Upload failed.";
        } else if (data.candidate_profile_status === "ready") {
          uploadMsg.textContent = data.candidate_profile_reused
            ? "Upload complete. Existing Candidate Profile reused."
            : "Upload complete. Candidate Profile created.";
        } else if (data.candidate_profile_status === "degraded") {
          uploadMsg.textContent = data.warning || "Upload complete. Local Candidate Profile created.";
        } else {
          uploadMsg.textContent = data.warning || "Resume saved, but Candidate Profile creation failed. Re-upload to retry.";
        }
        refreshStatus();
      })
      .catch((err) => {
        uploadMsg.textContent = err.message || "Upload failed.";
      })
      .finally(() => {
        uploadBtn.disabled = false;
        uploadBtn.textContent = "Choose file";
        resumeFile.disabled = false;
        resumeFile.value = "";
      });
  });
}

uploadBtn.addEventListener("click", () => {
  resumeFile.click();
});

resumeFile.addEventListener("change", () => {
  uploadSelectedResume();
});

analyzeBtn.addEventListener("click", () => {
  analysisIsRunning = true;
  updateAnalyzeAvailability();
  analyzeMsg.textContent = "Reading the current job page…";
  chrome.runtime.sendMessage({ type: "ANALYZE_CURRENT_TAB" }, (resp) => {
    if (resp && resp.in_progress) {
      renderAnalysisState(resp.state);
    } else if (resp && resp.ok) {
      analyzeMsg.textContent = "Reading the current job page…";
    } else {
      analysisIsRunning = false;
      updateAnalyzeAvailability();
      analyzeMsg.textContent = resp && resp.error ? resp.error : "Analyze failed.";
    }
  });
});

chrome.runtime.onMessage.addListener((msg) => {
  if (msg && msg.type === "AUTH_UPDATED") {
    refreshStatus();
    return;
  }
  if (msg && msg.type === "EXTRACTION_READY") {
    const meta = msg.data || {};
    const count = Number(meta.description_chars || meta.sent_chars || 0).toLocaleString();
    const source = String(meta.source || "page").replaceAll("_", " ");
    const role = [meta.title, meta.company, meta.location].filter(Boolean).join(" · ");
    analyzeMsg.textContent = `Extracted ${count} JD characters via ${source} (${meta.quality || "unknown"})${role ? ` — ${role}` : ""}. Analyzing...`;
    return;
  }
  if (msg && msg.type === "ANALYSIS_FINISHED") {
    analysisIsRunning = false;
    updateAnalyzeAvailability();
    if (msg.ok) {
      analyzeMsg.textContent = msg.data && msg.data.notion_url
        ? "Analysis saved to Notion."
        : "Analysis finished.";
    } else {
      analyzeMsg.textContent = msg.error || "Analysis failed.";
    }
  }
});

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName === "local" && changes[ANALYSIS_STATE_KEY]) {
    renderAnalysisState(changes[ANALYSIS_STATE_KEY].newValue);
  }
});

refreshStatus();
restoreAnalysisState();
