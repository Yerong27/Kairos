# Kairos AI Job Agent

> 中文说明在前，English documentation follows.

Kairos 是一个本地运行的求职匹配助手。Chrome extension 从当前 LinkedIn 职位页面提取结构化 JD；Gemini 分别解析简历和 JD；本地确定性评分器计算匹配度、seniority gap 和 `Should Apply?`；结果写入用户选择的 Notion 数据库。

## 核心设计

Kairos 将两个 AI 任务完全分开：

```text
上传新简历
  → Resume Parser（Gemini，仅在简历内容变化时调用）
  → Candidate Profile（本地 SQLite）

打开新职位
  → JD Parser（Gemini，只接收 JD，不接收简历）
  → Job Profile

Candidate Profile + Job Profile
  → 本地确定性评分
  → Score / Missing / Should Apply?
  → Notion
```

- 相同简历即使改名后重新上传，也会根据内容 hash 复用 Candidate Profile。
- 简历内容变化、Profile schema/prompt/model 变化时会重新解析。
- 每个新 JD 仍需要一次 Gemini 请求。
- 最终评分和 `Should Apply?` 不由 Gemini直接生成。

## 功能

- LinkedIn JSON-LD、Shadow DOM 和多级 selector 提取
- 自动展开职位描述并过滤相关推荐、公司介绍等页面噪音
- Resume 与 JD 使用独立 Gemini prompt、schema 和缓存生命周期
- JD 要求必须附带可在原文验证的 evidence quote
- MUST / SHOULD / NICE-TO-HAVE 要求分级
- Match / Partial / Missing 的完整要求矩阵
- 显式 JD 工具和 ATS 关键词覆盖
- 年限、近期经历、职责和 ownership 融合的 seniority 判断
- 确定性 score、score band 和 `Should Apply?`
- Notion OAuth 和数据库选择
- 不完整分析不缓存，也不写入 `Error (Degraded)` 页面

## 快速开始

### 1. 环境要求

- Python 3.10+
- Google Chrome 或 Chromium 浏览器
- Gemini API key
- Notion public integration
- 一个 Notion 数据库

### 2. 安装

```bash
git clone https://github.com/Yerong27/Kairos.git
cd Kairos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

### 3. 配置 `.env`

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_RESUME_MODEL=gemini-2.5-flash-lite

NOTION_CLIENT_ID=your_notion_client_id
NOTION_CLIENT_SECRET=your_notion_client_secret
NOTION_REDIRECT_URI=http://localhost:8000/notion/callback
```

在 Notion integration 中添加完全相同的 OAuth redirect URI：

```text
http://localhost:8000/notion/callback
```

不要提交 `.env`。本仓库只跟踪无密钥的 `.env.example`。

### 4. 启动后端

```bash
uvicorn main:app --reload
```

健康检查：<http://127.0.0.1:8000/health>

### 5. 安装 Chrome extension

1. 打开 `chrome://extensions/`。
2. 启用 Developer mode。
3. 点击 Load unpacked。
4. 选择仓库中的 `browser-extension` 文件夹。
5. 修改 extension 代码后，在该页面点击刷新。

### 6. 使用

1. 在 extension 中连接 Notion 并选择目标数据库。
2. 上传 PDF 或 TXT 简历。
3. 等待 `Candidate Profile created` 或 `Existing Candidate Profile reused`。
4. 打开 LinkedIn 职位详情页。
5. 点击 `Analyze this page`。
6. 在 Notion 查看完整匹配矩阵。

如果 Resume Parser 因额度或网络失败，简历文本仍会保存在本地，但分析按钮会保持禁用。重新上传同一文件即可重试；已成功的相同 Candidate Profile 不会重复调用 Gemini。

## Notion 数据库字段

推荐字段名称和类型：

| 字段 | 类型 | 要求 |
| --- | --- | --- |
| Job Title | Title | 必需 |
| Company | Text | 推荐 |
| Match Score | Number | 推荐 |
| Should Apply? | Select | 推荐 |
| Seniority Gap | Select | 推荐 |
| Missing Skills | Multi-select | 推荐 |
| Required Skills | Multi-select | 推荐 |
| URL | URL | 推荐 |

如果推荐字段不存在或类型错误，Kairos 会尝试 Safe Mode，仅写入标题和核心正文。

## 本地数据与隐私

Kairos 是本地工具，不是托管服务。以下数据保存在当前电脑：

```text
data/notion_oauth.db
  ├── Notion OAuth access token
  ├── 已选择的数据库
  ├── 提取后的简历文本
  └── Candidate Profile JSON

data/v3_cache/
  └── 职位分析缓存

debug_runs/（仅启用 KAIROS_DEBUG_DUMP 时）
  └── 调试结果
```

SQLite 中的 OAuth token、简历和 Candidate Profile 当前为明文。不要分享 `data/`、`.env` 或 `debug_runs/`。这些路径均已加入 `.gitignore`。

发送给外部服务的数据：

- 上传新简历时，简历文本发送给配置的 Gemini Resume Parser。
- 分析职位时，只有清洗后的 JD 发送给 Gemini JD Parser；简历不会随 JD 请求发送。
- 分析结果发送给用户授权的 Notion workspace。

## 缓存与更新

Candidate Profile 的有效性由以下组合决定：

```text
resume content hash
+ resume model
+ profile schema version
+ resume prompt version
```

任一项变化都会使旧 Profile 失效。职位缓存使用清洗后的 JD、Candidate Profile 版本、标题和输出语言生成 key；degraded 结果不会缓存。

## 测试

单元测试不会调用真实 Gemini 或 Notion：

```bash
python -m pytest tests -q
```

当前测试覆盖 OAuth 安全、Notion 重试、seniority、工具提取、要求矩阵、Candidate Profile hash 复用，以及 JD prompt 不包含简历。

## 常见问题

### Backend not reachable

确认 `uvicorn main:app --reload` 正在端口 8000 运行，并检查 <http://127.0.0.1:8000/health>。

### No databases found

Notion OAuth 后搜索索引可能短暂延迟。等待几秒后刷新选择页面，并确认目标数据库已分享给 Kairos integration。

### Candidate Profile is error / missing

Resume Parser 没有成功完成，常见原因是 Gemini `429 quota exceeded`、API key 无效或网络错误。重新上传同一简历会重试；成功的相同 Profile 会复用。

### Gemini 429

Gemini 限额按项目和模型计算。等待额度恢复或启用付费额度。Kairos 不会缓存失败结果或写入误导性的 Notion 页面。

### PDF 上传失败

当前 PDF 必须包含可提取文字；扫描图片 PDF 尚不支持 OCR。

## 当前限制

- LinkedIn DOM 可能变化，提取器使用多级 fallback，但仍可能需要更新 selector。
- LLM 提取具有不确定性；Kairos 使用原文验证和确定性评分限制其影响，但不能保证招聘结果或 ATS 通过。
- SQLite 数据尚未加密。
- 当前仅支持本机单用户式工作流，不适合直接暴露到公网。

---

# English

Kairos is a local-first job matching assistant. Its Chrome extension extracts a structured job description from the active LinkedIn page. Gemini parses resumes and job descriptions in separate workflows. A deterministic local engine calculates fit, seniority gap, and `Should Apply?`, then writes the result to a user-selected Notion database.

## Architecture

```text
New resume upload
  → Resume Parser (Gemini, only when resume content changes)
  → Candidate Profile (local SQLite)

New job page
  → JD Parser (Gemini receives the JD only, never the resume)
  → Job Profile

Candidate Profile + Job Profile
  → Deterministic local scoring
  → Score / Missing / Should Apply?
  → Notion
```

- Re-uploading identical resume content reuses the existing Candidate Profile, even if the filename changed.
- A changed resume, profile schema, prompt, or model triggers a new resume parse.
- Every new JD still requires one Gemini request.
- Gemini does not directly generate the final score or application recommendation.

## Features

- LinkedIn extraction through JSON-LD, Shadow DOM, dedicated selectors, and fallbacks
- Automatic job-description expansion and surrounding-page noise removal
- Separate Gemini prompts, schemas, and cache lifecycles for resumes and JDs
- JD-grounded requirement evidence
- MUST / SHOULD / NICE-TO-HAVE classification
- Complete Match / Partial / Missing requirement matrix
- Explicit JD tool and ATS-keyword coverage
- Seniority inference using dates, recent roles, responsibilities, and ownership
- Deterministic scores, bands, and `Should Apply?`
- Notion OAuth and database selection
- No caching or Notion writes for incomplete analyses

## Quick start

### Requirements

- Python 3.10+
- Google Chrome or another Chromium browser
- A Gemini API key
- A Notion public integration
- A Notion database

### Install

```bash
git clone https://github.com/Yerong27/Kairos.git
cd Kairos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Configure `.env`:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_RESUME_MODEL=gemini-2.5-flash-lite

NOTION_CLIENT_ID=your_notion_client_id
NOTION_CLIENT_SECRET=your_notion_client_secret
NOTION_REDIRECT_URI=http://localhost:8000/notion/callback
```

Add the same redirect URI to your Notion integration, then start the backend:

```bash
uvicorn main:app --reload
```

Open `chrome://extensions/`, enable Developer mode, choose Load unpacked, and select `browser-extension` from this repository.

In the extension:

1. Connect Notion and select a database.
2. Upload a PDF or TXT resume.
3. Wait for the Candidate Profile to become ready.
4. Open a LinkedIn job page.
5. Select `Analyze this page`.
6. Review the full match matrix in Notion.

## Recommended Notion properties

| Property | Type | Requirement |
| --- | --- | --- |
| Job Title | Title | Required |
| Company | Text | Recommended |
| Match Score | Number | Recommended |
| Should Apply? | Select | Recommended |
| Seniority Gap | Select | Recommended |
| Missing Skills | Multi-select | Recommended |
| Required Skills | Multi-select | Recommended |
| URL | URL | Recommended |

If optional properties are missing or invalid, Kairos attempts a safe-mode write with the title and core page body.

## Local data and privacy

Kairos is a local tool, not a hosted service. It stores Notion OAuth tokens, extracted resume text, and Candidate Profiles in `data/notion_oauth.db`. Job-analysis caches live in `data/v3_cache/`. Optional debug dumps live in `debug_runs/`.

These local records are currently stored in plaintext. Never share `.env`, `data/`, or `debug_runs/`. All are excluded from Git.

External data flow:

- On a new resume upload, extracted resume text is sent to the configured Gemini Resume Parser.
- During job analysis, only the cleaned JD is sent to the Gemini JD Parser. The resume is not included.
- Analysis results are sent to the Notion workspace authorized by the user.

## Candidate Profile lifecycle

A Candidate Profile is keyed by:

```text
resume content hash
+ resume model
+ profile schema version
+ resume prompt version
```

Any change invalidates the old profile. Re-uploading the same resume retries a failed profile but reuses a successful current profile.

## Tests

Unit tests do not call live Gemini or Notion services:

```bash
python -m pytest tests -q
```

## Troubleshooting

- **Backend unreachable:** confirm the backend is running on port 8000 and open <http://127.0.0.1:8000/health>.
- **No databases found:** wait briefly after OAuth, refresh the selection page, and confirm the database was shared with the integration.
- **Candidate Profile missing/error:** re-upload the resume. This retries failed parsing without re-parsing a successful identical profile.
- **Gemini 429:** wait for quota recovery or enable paid quota. Failed results are not cached or written to Notion.
- **PDF failure:** PDFs must contain extractable text. OCR is not currently supported.

## Limitations

- LinkedIn may change its DOM and require selector updates.
- LLM extraction is probabilistic. Grounded evidence and deterministic scoring reduce, but do not eliminate, model error.
- Local SQLite data is not encrypted.
- The backend is intended for a local single-user workflow and should not be exposed directly to the public internet.

## License

MIT. See [LICENSE](LICENSE).
