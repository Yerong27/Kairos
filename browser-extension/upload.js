const API_BASE = "http://127.0.0.1:8000";

const statusEl = document.getElementById("status");
const fileInput = document.getElementById("resume-file");
const uploadBtn = document.getElementById("upload-btn");
const uploadMsg = document.getElementById("upload-msg");

function getToken(cb) {
  chrome.storage.local.get(["user_token"], (result) => {
    const token = result && result.user_token ? String(result.user_token) : "";
    cb(token);
  });
}

getToken((token) => {
  statusEl.textContent = token ? "Ready to upload." : "Not connected. Please connect Notion first.";
});

uploadBtn.addEventListener("click", () => {
  uploadMsg.textContent = "";
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    uploadMsg.textContent = "Please choose a TXT or PDF file.";
    return;
  }
  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading and analyzing...";

  getToken((token) => {
    if (!token) {
      uploadMsg.textContent = "Not connected. Please connect Notion first.";
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
        if (!data || data.status !== "saved") {
          uploadMsg.textContent = "Upload failed.";
        } else if (data.candidate_profile_status === "ready") {
          uploadMsg.textContent = data.candidate_profile_reused
            ? "Upload complete. Existing Candidate Profile reused."
            : "Upload complete. Candidate Profile created.";
        } else {
          uploadMsg.textContent = data.warning || "Resume saved, but Candidate Profile creation failed. Re-upload to retry.";
        }
      })
      .catch((err) => {
        uploadMsg.textContent = err.message || "Upload failed.";
      })
      .finally(() => {
        uploadBtn.disabled = false;
        uploadBtn.textContent = "Upload";
      });
  });
});
