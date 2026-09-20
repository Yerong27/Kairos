import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const extensionRoot = new URL("../browser-extension/", import.meta.url);
const backgroundSource = readFileSync(new URL("background.js", extensionRoot), "utf8");
const contentSource = readFileSync(new URL("content.js", extensionRoot), "utf8");

function siteMatcher() {
  const addListener = () => {};
  const chrome = {
    runtime: { onInstalled: { addListener }, onStartup: { addListener }, onMessage: { addListener } },
    webNavigation: { onCompleted: { addListener } },
  };
  return vm.runInNewContext(`${backgroundSource}\nsupportedJobSite`, { chrome, URL, console });
}

function element(selector, text) {
  return {
    tagName: "DIV",
    innerText: text,
    textContent: text,
    children: [],
    shadowRoot: null,
    matches: (candidate) => candidate === selector,
  };
}

async function extractFromFixture({ url, title, elements = [], jsonLd = [], clock = Date }) {
  const messages = [];
  const document = {
    title,
    querySelectorAll: (selector) => selector === "*" ? elements :
      selector === 'script[type="application/ld+json"]' ? jsonLd : [],
    createElement: () => ({
      set innerHTML(value) { this.textContent = value.replace(/<[^>]+>/g, " "); },
      innerText: "",
      textContent: "",
    }),
  };
  const parsed = new URL(url);
  const chrome = { runtime: { sendMessage: (message, callback) => {
    messages.push(message);
    if (callback) callback({ ok: true });
  } } };
  vm.runInNewContext(contentSource, {
    window: { location: { href: url, hostname: parsed.hostname } },
    document,
    chrome,
    console,
    Date: clock,
    setTimeout,
  });
  await new Promise((resolve) => setImmediate(resolve));
  return messages;
}

test("only supported job detail URLs are accepted", () => {
  const supportedJobSite = siteMatcher();
  assert.equal(supportedJobSite("https://www.linkedin.com/jobs/view/123/"), "linkedin");
  assert.equal(supportedJobSite("https://www.seek.com.au/job/12345678?type=promoted"), "seek");
  assert.equal(supportedJobSite("https://seek.com.au/job/12345678"), "seek");
  assert.equal(supportedJobSite("https://www.seek.com.au/jobs-in-technology"), "");
  assert.equal(supportedJobSite("https://seek.com.au.evil.example/job/12345678"), "");
});

test("SEEK detail container is sent without surrounding search results", async () => {
  const description = "About the role: build reliable services and collaborate with users. ".repeat(12);
  const messages = await extractFromFixture({
    url: "https://www.seek.com.au/job/12345678?type=promoted",
    title: "Product Engineer at Example - SEEK",
    elements: [
      element("body", `Other jobs and navigation. ${"Unrelated listing. ".repeat(80)}`),
      element('[data-automation="jobAdDetails"]', description),
      element('[data-automation="job-detail-title"]', "Product Engineer"),
      element('[data-automation="advertiser-name"]', "Example"),
    ],
  });
  const payload = messages.find((message) => message.type === "JD_EXTRACT");
  assert.ok(payload);
  assert.equal(payload.title, "Product Engineer");
  assert.equal(payload.company, "Example");
  assert.equal(payload.url, "https://www.seek.com.au/job/12345678");
  assert.equal(payload.extraction_meta.site, "seek");
  assert.equal(payload.extraction_meta.source, "seek_selector");
  assert.ok(payload.page_text.includes("About the role"));
  assert.ok(!payload.page_text.includes("Unrelated listing"));
});

test("SEEK JobPosting JSON-LD works when its detail selector is absent", async () => {
  const description = "<p>About the role</p><p>Deliver customer research and product experiments.</p>".repeat(7);
  const script = { textContent: JSON.stringify({
    "@type": "JobPosting",
    title: "Product Researcher",
    hiringOrganization: { name: "Example Co" },
    description,
  }) };
  const messages = await extractFromFixture({
    url: "https://www.seek.com.au/job/87654321",
    title: "Product Researcher at Example Co - SEEK",
    jsonLd: [script],
  });
  const payload = messages.find((message) => message.type === "JD_EXTRACT");
  assert.ok(payload);
  assert.equal(payload.company, "Example Co");
  assert.equal(payload.extraction_meta.source, "json_ld");
  assert.ok(payload.page_text.includes("Deliver customer research"));
});

test("SEEK does not send an unreadable job page for AI analysis", async () => {
  let tick = 0;
  const clock = { now: () => { tick += 6000; return tick; } };
  const messages = await extractFromFixture({
    url: "https://www.seek.com.au/job/87654321",
    title: "Product Researcher at Example Co - SEEK",
    elements: [element("body", "Navigation and recommended jobs only.")],
    clock,
  });
  assert.equal(messages.some((message) => message.type === "JD_EXTRACT"), false);
  assert.equal(messages[0].type, "JD_EXTRACTION_FAILED");
  assert.match(messages[0].error, /SEEK job description was not found/);
});

test("LinkedIn extraction remains available", async () => {
  const description = "Build and operate a customer-facing product with the team. ".repeat(14);
  const messages = await extractFromFixture({
    url: "https://www.linkedin.com/jobs/view/12345678/",
    title: "Engineer | Example | LinkedIn",
    elements: [
      element("main h1", "Engineer"),
      element(".jobs-description__content .jobs-box__html-content", description),
    ],
  });
  const payload = messages.find((message) => message.type === "JD_EXTRACT");
  assert.ok(payload);
  assert.equal(payload.title, "Engineer");
  assert.equal(payload.extraction_meta.site, "linkedin");
  assert.equal(payload.extraction_meta.source, "linkedin_selector");
});
