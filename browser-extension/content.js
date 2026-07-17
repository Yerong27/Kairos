(() => {
  "use strict";

  const MAX_DESCRIPTION_CHARS = 18000;
  const END_MARKERS = [
    "About the company", "More jobs", "See more jobs", "People also viewed",
    "Similar jobs", "Set alert", "Report this job", "Meet the hiring team",
    "Show more jobs", "This job alert is on",
  ];
  const NAV_MARKERS = ["Skip to search", "Skip to main content", "My Network", "Notifications"];

  function clean(value) {
    return String(value || "")
      .replace(/\u00a0/g, " ")
      .replace(/\r/g, "")
      .replace(/[ \t]+\n/g, "\n")
      .replace(/[ \t]{2,}/g, " ")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  function allElements() {
    const result = [];
    const walk = (root) => {
      for (const element of root.querySelectorAll("*")) {
        result.push(element);
        if (element.shadowRoot) walk(element.shadowRoot);
      }
    };
    walk(document);
    return result;
  }

  function deepQueryAll(selector, pool) {
    const result = [];
    for (const element of (pool || allElements())) {
      try {
        if (element.matches(selector)) result.push(element);
      } catch (_error) {
        // LinkedIn occasionally leaves malformed transient selectors in the DOM.
      }
    }
    return result;
  }

  function firstText(selectors, pool) {
    for (const selector of selectors) {
      const element = deepQueryAll(selector, pool)[0];
      const text = clean(element && (element.innerText || element.textContent));
      if (text) return text;
    }
    return "";
  }

  function readJsonLd() {
    for (const script of document.querySelectorAll('script[type="application/ld+json"]')) {
      try {
        const parsed = JSON.parse(script.textContent || "null");
        const queue = Array.isArray(parsed) ? parsed : [parsed];
        for (const item of queue) {
          if (!item) continue;
          const type = item["@type"];
          if (type === "JobPosting" || (Array.isArray(type) && type.includes("JobPosting"))) return item;
          if (Array.isArray(item["@graph"])) queue.push(...item["@graph"]);
        }
      } catch (_error) {
        // Ignore unrelated or malformed structured data blocks.
      }
    }
    return null;
  }

  function htmlToText(value) {
    const node = document.createElement("div");
    node.innerHTML = String(value || "");
    return clean(node.innerText || node.textContent);
  }

  function trimDescription(value) {
    let text = clean(value);
    let cut = text.length;
    for (const marker of END_MARKERS) {
      const index = text.indexOf(marker);
      if (index > 250 && index < cut) cut = index;
    }
    text = text.slice(0, cut).replace(/\s*[.…]{1,3}\s*(more|see more|show more)\s*$/i, "").trim();
    return text.slice(0, MAX_DESCRIPTION_CHARS);
  }

  function expandDescription(pool) {
    const selectors = [
      "button.jobs-description__footer-button", "button.show-more-less-html__button--more",
      'button[aria-label*="see more" i]', 'button[aria-label*="show more" i]',
    ];
    for (const selector of selectors) {
      for (const button of deepQueryAll(selector, pool)) {
        const label = clean(button.innerText || button.getAttribute("aria-label"));
        if (/more|show|see/i.test(label)) {
          try { button.click(); } catch (_error) { /* best effort */ }
        }
      }
    }
  }

  function parseDocumentTitle() {
    let value = clean(document.title).replace(/\s*\|\s*LinkedIn\s*$/i, "").replace(/^\(\d+\)\s*/, "");
    const hiring = value.match(/^(.*?)\s+hiring\s+(.*?)\s+in\s+(.*)$/i);
    if (hiring) return { company: hiring[1], title: hiring[2], location: hiring[3] };
    const parts = value.split(" | ").map(clean).filter(Boolean);
    if (parts.length >= 2) return { title: parts.slice(0, -1).join(" | "), company: parts.at(-1), location: "" };
    return { title: value, company: "", location: "" };
  }

  function descriptionByHeading(pool) {
    const heading = pool.find((element) => /^About the (job|role|position|team)$/i.test(clean(element.innerText)));
    if (!heading) return "";
    let container = heading;
    for (let index = 0; index < 7 && container.parentElement; index += 1) {
      container = container.parentElement;
      const text = clean(container.innerText);
      if (text.length > 500 && !NAV_MARKERS.some((marker) => text.includes(marker))) {
        return trimDescription(text.replace(/^About the (job|role|position|team)\s*/i, ""));
      }
    }
    return "";
  }

  function descriptionByLongestBlock(pool) {
    const ignoredTags = new Set(["HTML", "BODY", "MAIN", "NAV", "HEADER", "FOOTER", "ASIDE", "SCRIPT", "STYLE"]);
    let best = "";
    for (const element of pool) {
      if (ignoredTags.has(element.tagName)) continue;
      const text = clean(element.innerText);
      if (text.length < 400 || text.length > 25000) continue;
      if (NAV_MARKERS.some((marker) => text.includes(marker))) continue;
      const childOwnsText = Array.from(element.children || []).some(
        (child) => clean(child.innerText).length >= text.length * 0.95,
      );
      if (!childOwnsText && text.length > best.length) best = text;
    }
    return trimDescription(best);
  }

  function locationFromJsonLd(job) {
    if (!job || !job.jobLocation) return "";
    const location = Array.isArray(job.jobLocation) ? job.jobLocation[0] : job.jobLocation;
    const address = location && location.address;
    if (!address) return "";
    return clean([address.addressLocality, address.addressRegion, address.addressCountry].filter(Boolean).join(", "));
  }

  function extract() {
    const initialPool = allElements();
    expandDescription(initialPool);
    const pool = allElements();
    const jsonLd = readJsonLd();
    const documentFields = parseDocumentTitle();

    const title = firstText([
      ".job-details-jobs-unified-top-card__job-title h1", ".job-details-jobs-unified-top-card__job-title",
      ".jobs-unified-top-card__job-title", "h1.top-card-layout__title", 'h1[class*="job-title"]', "main h1", "h1",
    ], pool) || clean(jsonLd && jsonLd.title) || documentFields.title;

    const organization = jsonLd && jsonLd.hiringOrganization;
    const company = firstText([
      ".job-details-jobs-unified-top-card__company-name a", ".job-details-jobs-unified-top-card__company-name",
      ".jobs-unified-top-card__company-name a", "a.topcard__org-name-link", 'a[class*="company-name"]', 'a[href*="/company/"]',
    ], pool) || clean(typeof organization === "string" ? organization : organization && organization.name) || documentFields.company;

    const location = firstText([
      ".job-details-jobs-unified-top-card__primary-description-container .tvm__text",
      ".job-details-jobs-unified-top-card__bullet", ".jobs-unified-top-card__bullet", ".topcard__flavor--bullet",
    ], pool) || locationFromJsonLd(jsonLd) || documentFields.location;

    let source = "linkedin_selector";
    let description = firstText([
      ".jobs-description__content .jobs-box__html-content", ".jobs-description-content__text",
      ".jobs-description__content", "#job-details", "article.jobs-description__container",
      ".jobs-box__html-content", ".show-more-less-html__markup", ".description__text",
    ], pool);
    if (!description && jsonLd && jsonLd.description) { description = htmlToText(jsonLd.description); source = "json_ld"; }
    if (!description) { description = descriptionByHeading(pool); source = "about_heading"; }
    if (!description) { description = descriptionByLongestBlock(pool); source = "longest_block_fallback"; }
    description = trimDescription(description);

    const canonicalUrl = clean((jsonLd && jsonLd.url) || window.location.href.split("?")[0]);
    const quality = description.length >= 800 ? "good" : (description.length >= 300 ? "partial" : "poor");
    const structuredText = clean([
      `Job title: ${title}`, `Company: ${company}`, `Location: ${location}`,
      "", "Job description:", description,
    ].join("\n"));

    return {
      url: canonicalUrl,
      title: clean(title),
      company: clean(company),
      location: clean(location),
      page_text: structuredText,
      extraction_meta: {
        source, quality, description_chars: description.length,
        sent_chars: structuredText.length, json_ld: Boolean(jsonLd), shadow_dom: true,
      },
    };
  }

  async function extractWhenReady(timeoutMs = 5000) {
    const deadline = Date.now() + timeoutMs;
    let data = extract();
    while ((!data.title || data.extraction_meta.quality === "poor") && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 300));
      data = extract();
    }
    return data;
  }

  extractWhenReady()
    .then((payload) => {
      chrome.runtime.sendMessage({ type: "JD_EXTRACT", ...payload }, (response) => {
        if (!response || !response.ok) console.error("Kairos extraction failed:", response && response.error);
      });
    })
    .catch((error) => console.error("Kairos LinkedIn extraction failed:", error));
})();
