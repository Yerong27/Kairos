# Kairos

Kairos 是一个本地运行的求职匹配助手：从 LinkedIn 读取职位描述，与简历进行比较，并把结果保存到 Notion。

**[复制 Notion 模板](https://lateral-band-b45.notion.site/Kairos-3a0fad1d15b3803f8581dd2b466dd40e)**

[English](#english)

## 准备

- Python 3.10+
- Google Chrome 或 Chromium 浏览器
- Gemini API key
- Notion integration

## 安装

```bash
git clone https://github.com/Yerong27/Kairos.git
cd Kairos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

## 配置

### 1. 准备 Notion 数据库

打开 [Kairos Notion 模板](https://lateral-band-b45.notion.site/Kairos-3a0fad1d15b3803f8581dd2b466dd40e)，点击 `Duplicate`，复制到自己的 Notion workspace。

建议直接使用模板。Kairos 依赖特定的数据库字段名称和类型，手动新建数据库容易导致 `Error (Degraded)`。

### 2. 创建 Notion integration

在 Notion Integrations 页面创建一个 public integration，并设置 OAuth redirect URI：

```text
http://localhost:8000/notion/callback
```

记录 integration 的 Client ID 和 Client Secret。

### 3. 填写 `.env`

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_RESUME_MODEL=gemini-2.5-flash-lite

NOTION_CLIENT_ID=your_notion_client_id
NOTION_CLIENT_SECRET=your_notion_client_secret
NOTION_REDIRECT_URI=http://localhost:8000/notion/callback
```

不要提交或分享 `.env`。

## 启动

```bash
uvicorn main:app --reload
```

浏览器打开 <http://127.0.0.1:8000/health>。看到 `ok` 即表示后端已启动。

## 安装 Chrome Extension

1. 打开 `chrome://extensions/`。
2. 开启右上角的 Developer mode。
3. 点击 Load unpacked。
4. 选择本项目中的 `browser-extension` 文件夹。

## 使用

1. 点击 Kairos Extension。
2. 连接 Notion，并在授权时选择刚复制的 `Kairos Job Tracker`。
3. 上传 PDF 或 TXT 简历，等待 Resume ready。
4. 打开 LinkedIn 职位详情页。
5. 点击 `Analyze this page`。
6. 在 Notion 查看匹配分数、`Should Apply?`、要求覆盖和简历证据。

更换简历时重新上传即可。相同内容的简历不会重复解析。

## 常见问题

### Backend not reachable

确认 `uvicorn main:app --reload` 仍在运行，并检查 <http://127.0.0.1:8000/health>。

### No databases found

确认 OAuth 授权时选择了复制后的数据库。首次连接后也可能需要等待几秒，再刷新数据库选择页面。

### Error (Degraded)

通常表示数据库字段不匹配。建议重新复制官方 Notion 模板并在 Extension 中切换到该数据库。

### Candidate Profile missing / error

重新上传简历。常见原因包括 Gemini API key 错误、免费额度耗尽或网络失败。

### Gemini 429

Gemini 免费额度已用完。等待额度恢复，或为对应的 Google AI 项目启用付费额度。

### PDF 无法读取

PDF 必须包含可选择的文字。扫描图片型 PDF 暂不支持 OCR，可先转换为 TXT。

## 隐私

Kairos 在本机保存 Notion 授权信息、简历文本和分析缓存。请勿分享 `.env`、`data/` 或 `debug_runs/`。不要把本地后端直接暴露到公网。

## 测试

```bash
python -m pytest tests -q
```

---

# English

Kairos is a local job-matching assistant. It reads a job description from LinkedIn, compares it with your resume, and saves the result to Notion.

**[Duplicate the Notion template](https://lateral-band-b45.notion.site/Kairos-3a0fad1d15b3803f8581dd2b466dd40e)**

## Requirements

- Python 3.10+
- Google Chrome or another Chromium browser
- A Gemini API key
- A Notion integration

## Install

```bash
git clone https://github.com/Yerong27/Kairos.git
cd Kairos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Configure

### 1. Prepare the Notion database

Open the [Kairos Notion template](https://lateral-band-b45.notion.site/Kairos-3a0fad1d15b3803f8581dd2b466dd40e) and select `Duplicate` to copy it into your workspace.

Using the template is recommended because Kairos expects specific property names and types. A manually created database may produce `Error (Degraded)`.

### 2. Create a Notion integration

Create a public integration from the Notion Integrations page and add this OAuth redirect URI:

```text
http://localhost:8000/notion/callback
```

Copy its Client ID and Client Secret.

### 3. Configure `.env`

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_RESUME_MODEL=gemini-2.5-flash-lite

NOTION_CLIENT_ID=your_notion_client_id
NOTION_CLIENT_SECRET=your_notion_client_secret
NOTION_REDIRECT_URI=http://localhost:8000/notion/callback
```

Never commit or share `.env`.

## Start Kairos

```bash
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000/health>. An `ok` response means the backend is ready.

## Install the Chrome extension

1. Open `chrome://extensions/`.
2. Enable Developer mode.
3. Select Load unpacked.
4. Choose the `browser-extension` folder from this repository.

## Use Kairos

1. Open the Kairos extension.
2. Connect Notion and authorize the duplicated `Kairos Job Tracker` database.
3. Upload a PDF or TXT resume and wait for Resume ready.
4. Open a LinkedIn job page.
5. Select `Analyze this page`.
6. Review the score, `Should Apply?`, requirement coverage, and resume evidence in Notion.

Upload again whenever your resume changes. Identical resume content is reused without another resume parse.

## Troubleshooting

### Backend not reachable

Confirm `uvicorn main:app --reload` is still running and check <http://127.0.0.1:8000/health>.

### No databases found

Confirm that you selected the duplicated database during OAuth. After the first connection, wait a few seconds and refresh the database-selection page.

### Error (Degraded)

This usually means the database properties do not match. Duplicate the official Notion template and select that database in the extension.

### Candidate Profile missing / error

Upload the resume again. Common causes include an invalid Gemini API key, exhausted quota, or a network failure.

### Gemini 429

The Gemini quota has been exhausted. Wait for it to reset or enable billing for the relevant Google AI project.

### PDF cannot be read

The PDF must contain selectable text. Scanned image PDFs are not currently supported; convert the resume to TXT first.

## Privacy

Kairos stores Notion authorization data, extracted resume text, and analysis caches locally. Never share `.env`, `data/`, or `debug_runs/`, and do not expose the local backend directly to the internet.

## Tests

```bash
python -m pytest tests -q
```

## License

MIT. See [LICENSE](LICENSE).
