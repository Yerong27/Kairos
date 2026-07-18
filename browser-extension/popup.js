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
  analyzeBtn.disabled = true;
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
          resumeStatus.textContent = profileReady ? "Resume ready" : "Profile needs attention";
          setTone(resumeDot, profileReady ? "success" : "error");
          const meta = [];
          if (data.resume_filename) meta.push(data.resume_filename);
          if (data.resume_uploaded_at) meta.push(new Date(data.resume_uploaded_at * 1000).toLocaleDateString());
          resumeMeta.textContent = meta.join(" • ");
          uploadBtn.disabled = false;
          analyzeBtn.disabled = !profileReady;
          analyzeMsg.textContent = profileReady
            ? ""
            : "Re-upload the resume to create or retry its Candidate Profile.";
        } else {
          resumeStatus.textContent = "No resume uploaded";
          setTone(resumeDot, "warning");
          resumeMeta.textContent = "PDF, DOCX, or TXT";
          uploadBtn.disabled = false;
          analyzeBtn.disabled = true;
          analyzeMsg.textContent = "";
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
        analyzeBtn.disabled = true;
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
  analyzeMsg.textContent = "Analyzing...";
  chrome.runtime.sendMessage({ type: "ANALYZE_CURRENT_TAB" }, (resp) => {
    if (resp && resp.ok) {
      analyzeMsg.textContent = "Request sent.";
    } else {
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
    if (msg.ok) {
      analyzeMsg.textContent = msg.data && msg.data.notion_url
        ? "Analysis saved to Notion."
        : "Analysis finished.";
    } else {
      analyzeMsg.textContent = msg.error || "Analysis failed.";
    }
  }
});

refreshStatus();
