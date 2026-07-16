(() => {
  const title = document.title || "";
  const url = window.location.href || "";

  const selectors = [
    '[data-automation="jobAdDetails"]',
    '[data-automation="jobAdDetailsContainer"]',
    '[data-automation="jobDetails"]',
    '[data-automation="jobDetailsContent"]',
    '[data-automation="job-ad-details"]',
    '.jobad-details',
    'article[role="main"]',
    'main article',
    'main',
    'article',
  ];

  const keywordHints = [
    "responsibilities",
    "requirements",
    "what you’ll be doing",
    "what you'll be doing",
    "about the role",
    "skills",
    "qualifications",
    "experience",
  ];

  function scoreText(t) {
    const low = t.toLowerCase();
    let score = 0;
    for (const k of keywordHints) {
      if (low.includes(k)) score += 3;
    }
    return score;
  }

  function pickBestText() {
    const hard = document.querySelector('[data-automation="jobAdDetails"]');
    if (hard && hard.innerText && hard.innerText.trim().length > 200) {
      return hard.innerText.trim();
    }
    const seen = new Set();
    const candidates = [];
    for (let index = 0; index < selectors.length; index += 1) {
      const sel = selectors[index];
      const el = document.querySelector(sel);
      if (el && el.innerText) {
        const txt = el.innerText.trim();
        if (txt && !seen.has(txt)) {
          seen.add(txt);
          candidates.push({ txt, priority: selectors.length - index });
        }
      }
    }
    const all = document.body && document.body.innerText ? document.body.innerText.trim() : "";
    if (all) candidates.push({ txt: all, priority: 0 });

    let best = "";
    let bestScore = -1;
    for (const candidate of candidates) {
      const txt = candidate.txt;
      const len = txt.length;
      if (len < 200) continue;
      const s = scoreText(txt) * 4000 + candidate.priority * 750 + Math.min(len, 20000) / 10;
      if (s > bestScore) {
        bestScore = s;
        best = txt;
      }
    }
    return best || all || "";
  }

  let text = pickBestText();

  // Preserve headings and bullet boundaries; the backend uses section context
  // to distinguish requirements from general page text.
  text = text
    .replace(/\r/g, "")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  if (text.length > 20000) {
    text = text.slice(0, 20000);
  }

  const payload = {
    url,
    title,
    page_text: text,
  };

  chrome.runtime.sendMessage(
    { type: "JD_EXTRACT", ...payload },
    (resp) => {
      if (resp && resp.ok) {
        console.log("JD Extractor success:", resp.data);
      } else {
        console.error("JD Extractor error:", resp && resp.error ? resp.error : "unknown");
      }
    }
  );
})();
