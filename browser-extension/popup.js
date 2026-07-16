const API_BASE = "http://127.0.0.1:8000";

const notionStatus = document.getElementById("notion-status");
const dbName = document.getElementById("db-name");
const resumeSection = document.getElementById("resume-section");
const analyzeSection = document.getElementById("analyze-section");
const resumeStatus = document.getElementById("resume-status");
const resumeMeta = document.getElementById("resume-meta");
const uploadBtn = document.getElementById("upload-btn");
const uploadMsg = document.getElementById("upload-msg");
const resumeFile = document.getElementById("resume-file");
const chooseFileBtn = document.getElementById("choose-file-btn");
const fileNameEl = document.getElementById("file-name");
const analyzeBtn = document.getElementById("analyze-btn");
const analyzeMsg = document.getElementById("analyze-msg");
const notionMsg = document.getElementById("notion-msg");

function getToken(cb) {
  chrome.storage.local.get(["user_token"], (result) => {
    const token = result && result.user_token ? String(result.user_token) : "";
    cb(token);
  });
}

function setStatusDisconnected() {
  notionStatus.textContent = "Not connected";
  dbName.textContent = "";
  resumeStatus.textContent = "Resume not available";
  resumeMeta.textContent = "";
  notionMsg.textContent = "Connect Notion to continue.";
  resumeSection.classList.add("hidden");
  analyzeSection.classList.add("hidden");
  uploadBtn.disabled = true;
  analyzeBtn.disabled = true;
  document.getElementById("change-db-btn").disabled = true;
}

function refreshStatus() {
  notionMsg.textContent = "Checking connection...";
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
        dbName.textContent = data.database_name ? `Database: ${data.database_name}` : "";
        document.getElementById("change-db-btn").disabled = false;
        notionMsg.textContent = data.notion_connected ? "Ready." : "Connect Notion to continue.";
        if (data.resume_present) {
          resumeStatus.textContent = "Resume uploaded";
          const meta = [];
          if (data.resume_filename) meta.push(data.resume_filename);
          if (data.resume_uploaded_at) meta.push(new Date(data.resume_uploaded_at * 1000).toLocaleString());
          resumeMeta.textContent = meta.join(" • ");
          resumeSection.classList.remove("hidden");
          analyzeSection.classList.remove("hidden");
          uploadBtn.disabled = false;
          analyzeBtn.disabled = false;
        } else {
          resumeStatus.textContent = "Resume not uploaded";
          resumeMeta.textContent = "";
          resumeSection.classList.remove("hidden");
          analyzeSection.classList.remove("hidden");
          uploadBtn.disabled = false;
          analyzeBtn.disabled = true;
          analyzeMsg.textContent = "Upload a resume before analyzing.";
        }
      })
      .catch(() => {
        notionStatus.textContent = "Connection error";
        notionMsg.textContent = "Backend not reachable at 127.0.0.1:8000.";
        resumeSection.classList.add("hidden");
        analyzeSection.classList.add("hidden");
        uploadBtn.disabled = true;
        analyzeBtn.disabled = true;
      });
  });
}

document.getElementById("connect-btn").addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "OPEN_NOTION_CONNECT" });
});

document.getElementById("change-db-btn").addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "OPEN_NOTION_CONNECT" });
});

document.getElementById("open-upload-btn").addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "OPEN_UPLOAD_PAGE" });
});

uploadBtn.addEventListener("click", () => {
  uploadMsg.textContent = "";
  const file = resumeFile.files && resumeFile.files[0];
  if (!file) {
    uploadMsg.textContent = "Please choose a TXT or PDF file.";
    return;
  }
  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading...";
  resumeFile.disabled = true;
  chooseFileBtn.disabled = true;

  getToken((token) => {
    if (!token) {
      uploadMsg.textContent = "Connect Notion first.";
      uploadBtn.disabled = false;
      uploadBtn.textContent = "Upload";
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
        uploadMsg.textContent = data && data.status === "saved" ? "Upload complete." : "Upload failed.";
        refreshStatus();
      })
      .catch((err) => {
        uploadMsg.textContent = err.message || "Upload failed.";
      })
      .finally(() => {
      uploadBtn.disabled = false;
      uploadBtn.textContent = "Upload";
      resumeFile.disabled = false;
      chooseFileBtn.disabled = false;
    });
  });
});

chooseFileBtn.addEventListener("click", () => {
  resumeFile.click();
});

resumeFile.addEventListener("change", () => {
  const file = resumeFile.files && resumeFile.files[0];
  fileNameEl.textContent = file ? file.name : "No file selected";
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
